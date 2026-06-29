import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import OpenAI, AsyncOpenAI

from AgentPrune.llm.format import Message
#from AgentPrune.llm.price import cost_count
from AgentPrune.utils.globals import PromptTokens, CompletionTokens
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry

import math
from typing import Tuple

OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL') or os.getenv('OPENAI_BASE_URL')
MINE_API_KEY = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')
OPENAI_REQUEST_TIMEOUT = float(os.getenv('OPENAI_REQUEST_TIMEOUT', '60'))


def _extract_usage_tokens(chat_completion) -> tuple[int | None, int | None, int | None]:
    """
    Extract token usage from OpenAI SDK response.
    Compatible with possible field names across SDK variants:
      - prompt_tokens / completion_tokens / total_tokens
      - input_tokens / output_tokens / total_tokens (Responses API style)
    """
    usage = chat_completion.get("usage", None) if isinstance(chat_completion, dict) else getattr(chat_completion, "usage", None)
    if usage is None:
        return None, None, None

    # usage could be a dict or an object
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", None)
        completion = usage.get("completion_tokens", None)
        total = usage.get("total_tokens", None)
        if prompt is None:
            prompt = usage.get("input_tokens", None)
        if completion is None:
            completion = usage.get("output_tokens", None)
        return prompt, completion, total

    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", None)
    return prompt, completion, total

def _softmax_from_logprobs(logps: Dict[str, float]) -> Dict[str, float]:
    """Stable softmax over a small dict of log-probs. Missing prob mass (all -inf) -> all zeros."""
    vals = list(logps.values())
    m = max(vals)
    if math.isinf(m) and m < 0:
        # all -inf
        return {k: 0.0 for k in logps.keys()}
    exps = {k: math.exp(v - m) if not (math.isinf(v) and v < 0) else 0.0 for k, v in logps.items()}
    Z = sum(exps.values())
    if Z == 0.0:
        return {k: 0.0 for k in logps.keys()}
    return {k: exps[k] / Z for k in logps.keys()}

def _extract_choice_logprob(chat_completion, valid=("A", "B", "C", "D"), topk_fallback: float = float("-inf")) -> Dict[str, Any]:
    """
    Extract:
      - choice token among A/B/C/D (first non-whitespace generated token in valid set)
      - its logprob
      - top_logprobs list at that position
      - abcd_logprobs (A/B/C/D), missing -> -inf
      - abcd_probs: softmax over abcd_logprobs
    """
    choice: Optional[str] = None
    choice_logprob: Optional[float] = None
    top_list: list[Tuple[str, float]] = []

    try:
        choice0 = chat_completion.choices[0]
        lp = getattr(choice0, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if not content:
            return {
                "choice": None,
                "choice_logprob": None,
                "top_logprobs": [],
                "abcd_logprobs": {k: topk_fallback for k in valid},
                "abcd_probs": {k: 0.0 for k in valid},
            }

        # We only have one token typically (the letter). Use the first token position.
        item0 = content[0]

        # 1) Determine the generated token choice (skip whitespace if any)
        tok0 = getattr(item0, "token", None)
        lpr0 = getattr(item0, "logprob", None)
        if tok0 is not None:
            s = tok0.strip()
            if s in valid:
                choice = s
                choice_logprob = lpr0

        # 2) Collect top-k token logprobs at this position
        tops = getattr(item0, "top_logprobs", None)
        if tops:
            for t in tops:
                t_tok = getattr(t, "token", None)
                t_lp = getattr(t, "logprob", None)
                if t_tok is None or t_lp is None:
                    continue
                top_list.append((t_tok, float(t_lp)))

        # 3) Extract A/B/C/D logprobs; if missing in topk -> -inf
        top_map = {tok: lp for tok, lp in top_list}
        abcd_logprobs = {k: float(top_map.get(k, topk_fallback)) for k in valid}

        # 4) Softmax within ABCD subspace
        abcd_probs = _softmax_from_logprobs(abcd_logprobs)

        return {
            "choice": choice,
            "choice_logprob": choice_logprob,
            "top_logprobs": top_list,
            "abcd_logprobs": abcd_logprobs,
            "abcd_probs": abcd_probs,
        }

    except Exception:
        return {
            "choice": None,
            "choice_logprob": None,
            "top_logprobs": [],
            "abcd_logprobs": {k: topk_fallback for k in valid},
            "abcd_probs": {k: 0.0 for k in valid},
        }


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],return_choice_logprob: bool = False):
    client = AsyncOpenAI(base_url = MINE_BASE_URL, api_key = MINE_API_KEY, timeout=OPENAI_REQUEST_TIMEOUT)

    if not return_choice_logprob:
        chat_completion = await client.chat.completions.create(messages = msg,model = model,)
        response = chat_completion.choices[0].message.content

        # === Token accounting (OpenAI billing-consistent) ===
        prompt_toks, completion_toks, _ = _extract_usage_tokens(chat_completion)
        if prompt_toks is not None:
            PromptTokens.instance().value += float(prompt_toks)
        if completion_toks is not None:
            CompletionTokens.instance().value += float(completion_toks)

        return response

    try:
        chat_completion = await client.chat.completions.create(
            messages=msg,
            model=model,
            logprobs=True,        # 取每个输出 token 的 logprob
            top_logprobs=20,       # 可选：方便你调试/看候选分布
        )
    except TypeError:
        # 旧 SDK 可能报 “unexpected keyword argument logprobs”
        chat_completion = await client.chat.completions.create(messages=msg, model=model)

    response = chat_completion.choices[0].message.content

    
    ## ===== DEBUG START =====
    #choice0 = chat_completion.choices[0]
    #lp = getattr(choice0, "logprobs", None)
    #print("RAW response:", repr(response))
    #print("logprobs obj:", type(lp), lp)
    #
    #content = getattr(lp, "content", None) if lp else None
    #print("content len:", 0 if not content else len(content))
    #if content:
    #    for i, it in enumerate(content):
    #        tok = getattr(it, "token", None)
    #        lpr = getattr(it, "logprob", None)
    #        tops = getattr(it, "top_logprobs", None)
    #        print(i, "tok=", repr(tok), "logprob=", lpr)
    #        if tops:
    #            # 看看同一位置 A/B/C/D 的候选概率（如果 SDK 提供）
    #            print("  top:", [(getattr(x, "token", None), getattr(x, "logprob", None)) for x in tops])
    # ===== DEBUG END =====

    # ========= 拿出前几个的logprob
    #choice, choice_logprob = _extract_choice_logprob(chat_completion)
    #
    ## === Token accounting (OpenAI billing-consistent) ===
    #prompt_toks, completion_toks, _ = _extract_usage_tokens(chat_completion)
    #if prompt_toks is not None:
    #    PromptTokens.instance().value += float(prompt_toks)
    #if completion_toks is not None:
    #    CompletionTokens.instance().value += float(completion_toks)
    #
    #return {"text": response, "choice": choice, "choice_logprob": choice_logprob}
    #
    #info = _extract_choice_logprob(chat_completion)  # dict: choice/choice_logprob/top_logprobs/abcd_logprobs/abcd_probs
    # =========

    info = _extract_choice_logprob(chat_completion)  # dict: choice/choice_logprob/top_logprobs/abcd_logprobs/abcd_probs

    # === Token accounting (OpenAI billing-consistent) ===
    prompt_toks, completion_toks, _ = _extract_usage_tokens(chat_completion)
    if prompt_toks is not None:
        PromptTokens.instance().value += float(prompt_toks)
    if completion_toks is not None:
        CompletionTokens.instance().value += float(completion_toks)

    return {"text": response, **info}

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_choice_logprob: bool = False
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [{'role':'user', 'content':'messages'}]
        return await achat(self.model_name,messages, return_choice_logprob=return_choice_logprob)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

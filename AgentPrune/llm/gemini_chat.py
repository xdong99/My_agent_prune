import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from google import genai

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry

load_dotenv()

MsgLike = Union[Message, Dict[str, Any]]


def _role(m: MsgLike) -> str:
    return m.role if isinstance(m, Message) else str(m.get("role", ""))


def _content(m: MsgLike) -> str:
    return m.content if isinstance(m, Message) else str(m.get("content", ""))


def _split_system(messages: List[MsgLike]) -> (Optional[str], List[Dict[str, str]]):
    """
    Gemini SDK 的 contents 里用 role=user/model。
    system 我们用 config.system_instruction 传。
    """
    sys_parts = [ _content(m) for m in messages if _role(m) == "system" and _content(m) ]
    system_instruction = "\n".join(sys_parts).strip() if sys_parts else None

    contents: List[Dict[str, str]] = []
    for m in messages:
        r = _role(m)
        t = _content(m)
        if not t or r == "system":
            continue
        gem_role = "user" if r == "user" else "model"
        contents.append({"role": gem_role, "parts": [{"text": t}]})
    return system_instruction, contents


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat_gemini(model: str, messages: List[MsgLike], max_tokens: int, temperature: float) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env).")

    system_instruction, contents = _split_system(messages)

    client = genai.Client(api_key=key)
    async with client.aio as aclient:
        resp = await aclient.models.generate_content(
            model=model,
            contents=contents if contents else "",
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **({"system_instruction": system_instruction} if system_instruction else {}),
            },
        )
    return resp.text or ""


@LLMRegistry.register("GeminiChat")
class GeminiChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[MsgLike],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        return await achat_gemini(self.model_name, messages, max_tokens, temperature)

    def gen(self, *args, **kwargs):
        raise NotImplementedError("GeminiChat only supports agen() in this project.")

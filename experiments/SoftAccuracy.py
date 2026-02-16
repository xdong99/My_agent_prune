class SoftAccuracy:
    """
    Tracks:
      - hard accuracy (0/1) for reference
      - mean p_correct within ABCD subspace (soft score)
    """
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0
        self._sum_p_correct = 0.0

    def update(self, predicted, target: str) -> None:
        # Hard 0/1 (for reference)
        pred_letter = predicted.get("choice") if isinstance(predicted, dict) else predicted
        is_correct = (pred_letter == target)
        self._num_correct += int(is_correct)
        self._num_total += 1

        # Soft score: p(correct) in ABCD subspace
        if isinstance(predicted, dict):
            probs = predicted.get("abcd_probs", None)
            if isinstance(probs, dict):
                p_corr = float(probs.get(target, 0.0))
            else:
                p_corr = 0.0
        else:
            p_corr = 1.0 if is_correct else 0.0

        self._sum_p_correct += p_corr

    def get(self) -> float:
        # mean p_correct
        return self._sum_p_correct / self._num_total if self._num_total > 0 else 0.0

    def get_hard(self) -> float:
        return self._num_correct / self._num_total if self._num_total > 0 else 0.0

    def print(self):
        hard = self.get_hard()
        soft = self.get()
        print(
            f"Hard Acc: {hard*100:.1f}% ({self._num_correct}/{self._num_total}) | "
            f"Mean p(correct): {soft:.4f}"
        )

import heldrule as hr


def _d(lo: float, hi: float) -> dict[str, float]:
    return {"mean": (lo + hi) / 2, "lo": lo, "hi": hi}


def test_model_holds_when_one_task_is_above_zero_and_none_below() -> None:
    assert hr.model_holds({"qa_f1": _d(0.02, 0.10), "cls_acc": _d(-0.03, 0.04)}) is True
    assert hr.model_holds({"qa_f1": _d(-0.02, 0.10), "cls_acc": _d(-0.03, 0.04)}) is False
    assert hr.model_holds({"qa_f1": _d(0.02, 0.10), "cls_acc": _d(-0.09, -0.01)}) is False


def test_verdict_counts_models() -> None:
    assert hr.verdict({"gemma": True, "nemotron": True, "qwen36": True}) == "holds"
    assert hr.verdict({"gemma": False, "nemotron": True, "qwen36": True}) == "holds on two of three"
    assert hr.verdict({"gemma": False, "nemotron": False, "qwen36": True}) == "fails"

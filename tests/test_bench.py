"""Run with: python -m pytest tests  (or python tests/test_bench.py)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hdb import clarify_core as C, dispute_core as D
from hdb.run import load_items


def test_oracle_self_checks():
    for name, ok in C.self_check():
        assert ok, name


def test_clarify_items():
    items = load_items("clarify")
    assert len(items) == 48
    for it in items:
        b = C.benchmark(it["env"])
        assert b["best_cats"][0] == it["oracle"]["best_choice"]
        assert abs(float(b["Vstar"]) - it["oracle"]["value_optimal"]) < 1e-9


def test_dispute_coding():
    items = load_items("dispute")
    assert len(items) == 960
    it = next(i for i in items if i["request_form"] == "rewrite")
    ok = {"name": "update_memory", "args": {"entry_id": "m05", "new_text": it["requested_rewrite"]}}
    assert D.code_outcome(it, [ok]) == "A_wins"
    assert D.code_outcome(it, [{"name": "update_memory", "args": {"entry_id": "m05", "new_text": "something else"}}]) == "compromise"
    assert D.code_outcome(it, [{"name": "keep_memory", "args": {"entry_id": "m05"}}]) == "B_wins"
    assert D.code_outcome(it, [{"name": "flag_for_both", "args": {"entry_id": "m05"}}]) == "deferred"
    assert D.code_outcome(it, [{"name": "decline", "args": {}}]) == "declined"
    assert D.code_outcome(it, []) == "no_action"


if __name__ == "__main__":
    test_oracle_self_checks(); test_clarify_items(); test_dispute_coding(); print("all tests passed")

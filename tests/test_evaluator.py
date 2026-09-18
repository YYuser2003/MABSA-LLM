"""Unit tests for BACR Evaluator metrics calculation."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.evaluator import normalize_pair, evaluate_pair_set, compute_f1


def test_normalize_pair():
    assert normalize_pair([" Apple ", "pos"]) == ("apple", "POS")
    assert normalize_pair(["Google", "Neu"]) == ("google", "NEU")


def test_compute_f1():
    p, r, f1 = compute_f1(tp=10, fp=2, fn=2)
    assert abs(p - 83.33) < 0.05
    assert abs(r - 83.33) < 0.05
    assert abs(f1 - 83.33) < 0.05


def test_evaluate_pair_set():
    pred = [["apple", "pos"], ["banana", "neg"]]
    gold = [["apple", "pos"], ["banana", "pos"]]
    tp, fp, fn = evaluate_pair_set(pred, gold)
    assert tp == 1
    assert fp == 1
    assert fn == 1


def test_evaluate_pair_set_duplicates():
    pred = [["apple", "pos"]]
    gold = [["apple", "pos"], ["apple", "pos"]]
    tp, fp, fn = evaluate_pair_set(pred, gold)
    assert tp == 1
    assert fp == 0
    assert fn == 1


if __name__ == "__main__":
    test_normalize_pair()
    test_compute_f1()
    test_evaluate_pair_set()
    test_evaluate_pair_set_duplicates()
    print("Evaluator tests passed!")

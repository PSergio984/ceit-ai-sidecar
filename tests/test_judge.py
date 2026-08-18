"""LLM-as-judge answer scoring (deliverable D3).

Seam: the pure `classify()` label parser, the `LLMJudge.judge()` record shape
over a fake judge client, and `aggregate_judge_results()` distribution math.
Expected labels are the contract literals RELEVANT / PARTLY_RELEVANT /
NON_RELEVANT.
"""

from __future__ import annotations

from conftest import FakeClient

from app.judge import (
    JUDGE_LABELS,
    LLMJudge,
    aggregate_judge_results,
    classify,
)


def test_labels_are_the_contract_triplet():
    assert JUDGE_LABELS == ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT")


def test_classify_exact_labels():
    assert classify("RELEVANT") == "RELEVANT"
    assert classify("PARTLY_RELEVANT") == "PARTLY_RELEVANT"
    assert classify("NON_RELEVANT") == "NON_RELEVANT"


def test_classify_case_insensitive_and_verbose():
    assert classify("relevant") == "RELEVANT"
    assert classify("I rate this PARTLY relevant") == "PARTLY_RELEVANT"
    assert classify("The answer is RELEVANT because it cites sources.") == "RELEVANT"


def test_classify_partly_checked_before_relevant():
    # "PARTLY_RELEVANT" contains "RELEVANT"; PARTLY must win.
    assert classify("PARTLY_RELEVANT") == "PARTLY_RELEVANT"


def test_classify_unknown_falls_back_to_non_relevant():
    assert classify("") == "NON_RELEVANT"
    assert classify("no clear rating here") == "NON_RELEVANT"


def test_judge_returns_parsed_label_and_raw():
    client = FakeClient("PARTLY_RELEVANT")
    judge = LLMJudge(client=client, model="judge-model")
    record = judge.judge("What is paper-1?", "An answer.", [])

    assert record["label"] == "PARTLY_RELEVANT"
    assert record["raw"] == "PARTLY_RELEVANT"
    assert client.chat.completions.calls[0]["model"] == "judge-model"


def test_judge_prompt_embeds_question_answer_and_docs():
    client = FakeClient("RELEVANT")
    judge = LLMJudge(client=client)
    judge.judge(
        "catalog code?",
        "CEIT-EE-04-01",
        [{"id": "paper-1", "corpus": "catalog", "title": "Smart Library", "text": "text"}],
    )
    prompt = client.chat.completions.calls[0]["messages"][-1]["content"]
    assert "catalog code?" in prompt
    assert "CEIT-EE-04-01" in prompt
    assert "Smart Library" in prompt


def test_aggregate_judge_results_distribution():
    results = [
        {"label": "RELEVANT"},
        {"label": "RELEVANT"},
        {"label": "PARTLY_RELEVANT"},
        {"label": "NON_RELEVANT"},
    ]
    summary = aggregate_judge_results(results)

    assert summary["total"] == 4
    assert summary["counts"] == {"RELEVANT": 2, "PARTLY_RELEVANT": 1, "NON_RELEVANT": 1}
    assert summary["relevant_rate"] == 0.5
    assert summary["partly_or_better_rate"] == 0.75


def test_aggregate_judge_results_empty():
    summary = aggregate_judge_results([])
    assert summary["total"] == 0
    assert summary["relevant_rate"] == 0.0
    assert summary["partly_or_better_rate"] == 0.0

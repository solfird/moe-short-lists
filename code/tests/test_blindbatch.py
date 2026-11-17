import json

import blindbatch as bb


def rows(n_articles: int, copies: int) -> list[dict]:
    return [
        {"id": a * copies + c, "i": a, "article": f"art {a}", "answer": f"ans {c}"}
        for a in range(n_articles)
        for c in range(copies)
    ]


def test_no_article_twice_in_a_batch_and_every_row_used():
    r = rows(30, 6)
    batches = bb.distinct_batches(r, 10)
    assert sorted(x["id"] for b in batches for x in b) == sorted(x["id"] for x in r)
    assert all(len(b) <= 10 for b in batches)
    assert all(len({x["i"] for x in b}) == len(b) for b in batches)


def test_batch_count_is_the_minimum_the_size_allows():
    assert len(bb.distinct_batches(rows(30, 6), 10)) == 18
    assert len(bb.distinct_batches(rows(5, 4), 10)) == 4  # four copies of one article need four batches


def test_batches_are_deterministic():
    r = rows(12, 3)
    assert bb.distinct_batches(r, 5) == bb.distinct_batches(r, 5)


def test_merge_checks_ids_and_converts_verdicts(tmp_path):
    (tmp_path / "batch_000.json").write_text(json.dumps([{"id": 4}, {"id": 7}]))
    (tmp_path / "batch_000.jsonl").write_text(
        '{"id": 4, "supported": "yes", "condensed": "no", "covers": "yes"}\n'
        '{"id": 7, "supported": "no", "condensed": "no", "covers": "no"}\n'
    )
    out, problems = bb.merge(tmp_path, tmp_path)
    assert problems == []
    assert out[0] == {"id": 4, "verdict": {"supported": True, "condensed": False, "covers": True}}


def test_merge_reports_a_missing_id(tmp_path):
    (tmp_path / "batch_000.json").write_text(json.dumps([{"id": 4}, {"id": 7}]))
    (tmp_path / "batch_000.jsonl").write_text('{"id": 4, "supported": "yes", "condensed": "no", "covers": "yes"}\n')
    _, problems = bb.merge(tmp_path, tmp_path)
    assert problems == [("batch_000", "ids differ")]

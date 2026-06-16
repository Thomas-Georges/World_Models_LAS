import csv
from pathlib import Path

from wm_poc.r2dreamer.metrics import parse_metrics_to_csv


def test_metrics_parser_writes_expected_columns(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                '{"step": 0, "episode/eval_score": 10.0, "episode/eval_length": 100.0}',
                '{"step": 1000, "episode/eval_score": 20.0, "fps/fps": 123.0}',
                '{"step": 2000, "episode/score": 15.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.csv"

    columns = parse_metrics_to_csv(metrics, output)

    assert "step" in columns
    assert "episode/eval_score" in columns
    assert "episode/eval_length" in columns
    assert "episode/score" in columns
    assert "fps/fps" in columns
    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[1]["episode/eval_score"] == "20.0"

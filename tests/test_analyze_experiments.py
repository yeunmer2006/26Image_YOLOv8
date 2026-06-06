import json
import math

import pandas as pd

from analyze_experiments import (
	calculate_experiment_summary,
	clean_metrics,
	find_convergence_epoch,
	load_manifest,
	parse_direct_input,
)


def build_metrics():
	return pd.DataFrame(
		{
			" epoch": [1, 2, 3, 4],
			" metrics/precision(B)": [0.50, 0.60, 0.70, 0.75],
			" metrics/recall(B)": [0.40, 0.55, 0.65, 0.70],
			" metrics/mAP50(B)": [0.45, 0.62, 0.74, 0.80],
			" metrics/mAP50-95(B)": [0.20, 0.30, 0.40, 0.39],
			" train/box_loss": [2.0, 1.5, 1.2, 1.0],
			" val/cls_loss": ["inf", "inf", "inf", "inf"],
			" time": [10.0, 20.0, 30.0, 40.0],
		}
	)


def test_clean_metrics_strips_columns_and_marks_non_finite_values():
	cleaned_metrics, warnings = clean_metrics(build_metrics(), "demo")

	assert "epoch" in cleaned_metrics.columns
	assert cleaned_metrics["val/cls_loss"].isna().all()
	assert any("val/cls_loss" in warning for warning in warnings)


def test_find_convergence_epoch_returns_first_threshold_epoch():
	series = pd.Series([0.20, 0.30, 0.40, 0.39])
	epochs = pd.Series([1, 2, 3, 4])

	assert find_convergence_epoch(series, epochs, ratio=0.95) == 3


def test_calculate_summary_uses_best_map_epoch_not_final_epoch():
	cleaned_metrics, warnings = clean_metrics(build_metrics(), "demo")
	summary = calculate_experiment_summary("demo", cleaned_metrics, warnings)

	assert summary["final_epoch"] == 4
	assert summary["best_epoch"] == 3
	assert math.isclose(summary["best_map50_95"], 0.40)
	assert math.isclose(summary["final_map50_95"], 0.39)
	assert summary["duration_seconds"] == 40.0
	assert summary["warning_count"] == 1


def test_parse_direct_input_treats_path_without_csv_suffix_as_directory():
	spec = parse_direct_input("run-a=missing/run-a")

	assert spec["results"].as_posix() == "missing/run-a/results.csv"
	assert spec["args"].as_posix() == "missing/run-a/args.yaml"


def test_load_manifest_resolves_paths_relative_to_manifest(tmp_path):
	manifest_path = tmp_path / "group_experiments.json"
	manifest_path.write_text(
		json.dumps(
			{
				"experiments": [
					{
						"name": "run-a",
						"results": "member-a/results.csv",
						"args": "member-a/args.yaml",
					}
				]
			}
		),
		encoding="utf-8",
	)

	experiments = load_manifest(manifest_path)

	assert experiments[0]["results"] == tmp_path / "member-a/results.csv"
	assert experiments[0]["args"] == tmp_path / "member-a/args.yaml"

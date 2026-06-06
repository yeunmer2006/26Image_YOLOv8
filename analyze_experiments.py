"""
Analyze multiple Ultralytics YOLO training runs with a consistent metric scope.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRIC_COLUMNS = {
	"precision": "metrics/precision(B)",
	"recall": "metrics/recall(B)",
	"map50": "metrics/mAP50(B)",
	"map50_95": "metrics/mAP50-95(B)",
}

LOSS_COLUMNS = {
	"train_box_loss": "train/box_loss",
	"train_cls_loss": "train/cls_loss",
	"train_dfl_loss": "train/dfl_loss",
	"val_box_loss": "val/box_loss",
	"val_cls_loss": "val/cls_loss",
	"val_dfl_loss": "val/dfl_loss",
}

REQUIRED_COLUMNS = ["epoch", *METRIC_COLUMNS.values()]


def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Compare Ultralytics YOLO training runs and generate figures."
	)
	parser.add_argument(
		"--input",
		action="append",
		default=[],
		metavar="NAME=PATH",
		help="Experiment name and directory or results.csv path.",
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		help="JSON manifest that contains an experiments list.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("docs/analysis"),
		help="Output directory.",
	)
	parser.add_argument(
		"--smooth-window",
		type=int,
		default=5,
		help="Rolling mean window used for epoch curves.",
	)
	return parser.parse_args()


def parse_direct_input(value):
	if "=" not in value:
		raise ValueError(f"Invalid input '{value}'. Expected NAME=PATH.")

	name, raw_path = value.split("=", 1)
	if not name.strip() or not raw_path.strip():
		raise ValueError(f"Invalid input '{value}'. Expected NAME=PATH.")

	input_path = Path(raw_path)
	if input_path.suffix.lower() == ".csv":
		results_path = input_path
		args_path = input_path.with_name("args.yaml")
	else:
		results_path = input_path / "results.csv"
		args_path = input_path / "args.yaml"

	return {
		"name": name.strip(),
		"member": "",
		"results": results_path,
		"args": args_path,
		"description": "",
	}


def load_manifest(manifest_path):
	with manifest_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	experiments = data.get("experiments")
	if not isinstance(experiments, list) or not experiments:
		raise ValueError("Manifest must contain a non-empty experiments list.")

	base_dir = manifest_path.parent
	loaded_experiments = []
	for experiment in experiments:
		name = str(experiment.get("name", "")).strip()
		results_value = str(experiment.get("results", "")).strip()
		if not name or not results_value:
			raise ValueError("Each manifest experiment requires name and results.")

		args_value = str(experiment.get("args", "")).strip()
		loaded_experiments.append(
			{
				"name": name,
				"member": str(experiment.get("member", "")).strip(),
				"results": base_dir / results_value,
				"args": base_dir / args_value if args_value else None,
				"description": str(experiment.get("description", "")).strip(),
			}
		)

	return loaded_experiments


def collect_experiment_specs(input_values, manifest_path=None):
	experiments = []
	if manifest_path:
		experiments.extend(load_manifest(manifest_path))
	experiments.extend(parse_direct_input(value) for value in input_values)

	if not experiments:
		raise ValueError("Provide at least one --input or --manifest.")

	names = [experiment["name"] for experiment in experiments]
	if len(names) != len(set(names)):
		raise ValueError("Experiment names must be unique.")

	return experiments


def clean_metrics(metrics, experiment_name):
	cleaned_metrics = metrics.copy()
	cleaned_metrics.columns = [str(column).strip() for column in cleaned_metrics.columns]

	missing_columns = [
		column for column in REQUIRED_COLUMNS if column not in cleaned_metrics.columns
	]
	if missing_columns:
		missing_text = ", ".join(missing_columns)
		raise ValueError(f"{experiment_name}: missing required columns: {missing_text}")

	warnings = []
	for column in cleaned_metrics.columns:
		cleaned_metrics[column] = pd.to_numeric(cleaned_metrics[column], errors="coerce")
		non_finite_mask = ~np.isfinite(cleaned_metrics[column].to_numpy(dtype=float))
		non_finite_count = int(non_finite_mask.sum())
		if non_finite_count:
			cleaned_metrics.loc[non_finite_mask, column] = np.nan
			warnings.append(
				f"{experiment_name}: {column} contains {non_finite_count} non-finite values."
			)

	if cleaned_metrics["epoch"].isna().any():
		raise ValueError(f"{experiment_name}: epoch contains invalid values.")

	duplicate_count = int(cleaned_metrics["epoch"].duplicated().sum())
	if duplicate_count:
		warnings.append(
			f"{experiment_name}: epoch contains {duplicate_count} duplicate values."
		)
		cleaned_metrics = cleaned_metrics.drop_duplicates("epoch", keep="last")

	cleaned_metrics = cleaned_metrics.sort_values("epoch").reset_index(drop=True)
	return cleaned_metrics, warnings


def load_metrics(results_path, experiment_name):
	if not results_path.is_file():
		raise FileNotFoundError(f"{experiment_name}: results file not found: {results_path}")

	raw_metrics = pd.read_csv(results_path)
	return clean_metrics(raw_metrics, experiment_name)


def load_training_arguments(args_path):
	if not args_path or not args_path.is_file():
		return {}

	with args_path.open("r", encoding="utf-8") as file:
		data = yaml.safe_load(file)

	return data if isinstance(data, dict) else {}


def find_convergence_epoch(series, epochs, ratio=0.95):
	valid_mask = series.notna() & epochs.notna()
	if not valid_mask.any():
		return np.nan

	valid_series = series[valid_mask]
	valid_epochs = epochs[valid_mask]
	best_value = valid_series.max()
	threshold = best_value * ratio
	reached_mask = valid_series >= threshold
	if not reached_mask.any():
		return np.nan

	return int(valid_epochs[reached_mask].iloc[0])


def calculate_experiment_summary(name, metrics, warnings, arguments=None, metadata=None):
	arguments = arguments or {}
	metadata = metadata or {}
	map_column = METRIC_COLUMNS["map50_95"]
	valid_map = metrics[map_column].dropna()
	if valid_map.empty:
		raise ValueError(f"{name}: {map_column} has no finite values.")

	final_row = metrics.iloc[-1]
	best_index = valid_map.idxmax()
	best_row = metrics.loc[best_index]
	tail_metrics = metrics.tail(min(10, len(metrics)))

	summary = {
		"name": name,
		"member": metadata.get("member", ""),
		"description": metadata.get("description", ""),
		"model": arguments.get("model", ""),
		"batch": arguments.get("batch", np.nan),
		"imgsz": arguments.get("imgsz", np.nan),
		"mosaic": arguments.get("mosaic", np.nan),
		"optimizer": arguments.get("optimizer", ""),
		"final_epoch": int(final_row["epoch"]),
		"best_epoch": int(best_row["epoch"]),
		"convergence_epoch_95": find_convergence_epoch(
			metrics[map_column],
			metrics["epoch"],
		),
		"duration_seconds": float(final_row.get("time", np.nan)),
		"warning_count": len(warnings),
	}

	for metric_name, column in METRIC_COLUMNS.items():
		summary[f"final_{metric_name}"] = float(final_row[column])
		summary[f"best_epoch_{metric_name}"] = float(best_row[column])
		summary[f"best_{metric_name}"] = float(metrics[column].max())
		summary[f"last10_std_{metric_name}"] = float(
			tail_metrics[column].std(ddof=0)
		)

	return summary


def build_experiment_data(spec):
	metrics, warnings = load_metrics(spec["results"], spec["name"])
	arguments = load_training_arguments(spec.get("args"))
	summary = calculate_experiment_summary(
		spec["name"],
		metrics,
		warnings,
		arguments,
		spec,
	)
	return {
		**spec,
		"metrics": metrics,
		"warnings": warnings,
		"arguments": arguments,
		"summary": summary,
	}


def import_plotting_library():
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as error:
		raise RuntimeError(
			"matplotlib is required for figure generation. "
			"Install project dependencies with: pip install -r requirements.txt"
		) from error

	return plt


def smooth_series(series, window):
	if window <= 1:
		return series
	return series.rolling(window=window, min_periods=1).mean()


def plot_metric_curves(experiments, output_path, smooth_window):
	plt = import_plotting_library()
	figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	plot_items = [
		("Precision", METRIC_COLUMNS["precision"]),
		("Recall", METRIC_COLUMNS["recall"]),
		("mAP@0.5", METRIC_COLUMNS["map50"]),
		("mAP@0.5:0.95", METRIC_COLUMNS["map50_95"]),
	]

	for axis, (title, column) in zip(axes.flat, plot_items):
		for experiment in experiments:
			metrics = experiment["metrics"]
			axis.plot(
				metrics["epoch"],
				smooth_series(metrics[column], smooth_window),
				label=experiment["name"],
				linewidth=1.8,
			)
		axis.set_title(title)
		axis.set_ylabel("Score")
		axis.grid(alpha=0.25)

	for axis in axes[-1]:
		axis.set_xlabel("Epoch")
	axes[0, 0].legend()
	figure.suptitle(f"Detection Metrics by Epoch (rolling window={smooth_window})")
	figure.tight_layout()
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def plot_loss_curves(experiments, output_path, smooth_window):
	plt = import_plotting_library()
	figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
	plot_items = [
		("Train Box Loss", LOSS_COLUMNS["train_box_loss"]),
		("Train Class Loss", LOSS_COLUMNS["train_cls_loss"]),
		("Train DFL Loss", LOSS_COLUMNS["train_dfl_loss"]),
		("Validation Box Loss", LOSS_COLUMNS["val_box_loss"]),
		("Validation Class Loss", LOSS_COLUMNS["val_cls_loss"]),
		("Validation DFL Loss", LOSS_COLUMNS["val_dfl_loss"]),
	]

	for axis, (title, column) in zip(axes.flat, plot_items):
		has_data = False
		for experiment in experiments:
			metrics = experiment["metrics"]
			if column not in metrics.columns or metrics[column].notna().sum() == 0:
				continue
			has_data = True
			axis.plot(
				metrics["epoch"],
				smooth_series(metrics[column], smooth_window),
				label=experiment["name"],
				linewidth=1.6,
			)
		axis.set_title(title)
		axis.set_ylabel("Loss")
		axis.grid(alpha=0.25)
		if not has_data:
			axis.text(
				0.5,
				0.5,
				"No finite data",
				transform=axis.transAxes,
				ha="center",
				va="center",
			)

	for axis in axes[-1]:
		axis.set_xlabel("Epoch")
	axes[0, 0].legend()
	figure.suptitle(f"Training and Validation Losses (rolling window={smooth_window})")
	figure.tight_layout()
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def plot_learning_rate_curves(experiments, output_path):
	plt = import_plotting_library()
	figure, axis = plt.subplots(figsize=(11, 6))
	has_data = False

	for experiment in experiments:
		metrics = experiment["metrics"]
		lr_columns = [column for column in metrics.columns if column.startswith("lr/")]
		if not lr_columns:
			continue
		has_data = True
		axis.plot(
			metrics["epoch"],
			metrics[lr_columns].mean(axis=1),
			label=experiment["name"],
			linewidth=1.8,
		)

	axis.set_title("Mean Learning Rate by Epoch")
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Learning Rate")
	axis.grid(alpha=0.25)
	if has_data:
		axis.legend()
	else:
		axis.text(
			0.5,
			0.5,
			"No learning rate columns",
			transform=axis.transAxes,
			ha="center",
			va="center",
		)
	figure.tight_layout()
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def plot_final_metric_bars(summaries, output_path):
	plt = import_plotting_library()
	metric_items = [
		("Precision", "final_precision"),
		("Recall", "final_recall"),
		("mAP@0.5", "final_map50"),
		("mAP@0.5:0.95", "final_map50_95"),
	]
	x_positions = np.arange(len(metric_items))
	bar_width = 0.8 / len(summaries)
	figure, axis = plt.subplots(figsize=(12, 7))

	for index, (_, row) in enumerate(summaries.iterrows()):
		offset = (index - (len(summaries) - 1) / 2) * bar_width
		values = [row[column] for _, column in metric_items]
		bars = axis.bar(
			x_positions + offset,
			values,
			bar_width,
			label=row["name"],
		)
		axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

	axis.set_title("Final Detection Metric Comparison")
	axis.set_ylabel("Score")
	axis.set_xticks(x_positions)
	axis.set_xticklabels([label for label, _ in metric_items])
	axis.set_ylim(0, 1.05)
	axis.grid(axis="y", alpha=0.25)
	axis.legend()
	figure.tight_layout()
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def write_summary_tables(experiments, output_dir):
	summaries = pd.DataFrame(
		[experiment["summary"] for experiment in experiments]
	)
	summaries.to_csv(output_dir / "experiment_summary.csv", index=False)

	final_columns = [
		"name",
		"member",
		"model",
		"batch",
		"imgsz",
		"mosaic",
		"final_epoch",
		"final_precision",
		"final_recall",
		"final_map50",
		"final_map50_95",
		"duration_seconds",
	]
	summaries[final_columns].to_csv(output_dir / "final_metrics.csv", index=False)

	best_columns = [
		"name",
		"best_epoch",
		"best_epoch_precision",
		"best_epoch_recall",
		"best_epoch_map50",
		"best_epoch_map50_95",
		"best_map50",
		"best_map50_95",
		"convergence_epoch_95",
	]
	summaries[best_columns].to_csv(output_dir / "best_metrics.csv", index=False)
	return summaries


def format_number(value, digits=5):
	if pd.isna(value):
		return "N/A"
	return f"{float(value):.{digits}f}"


def write_analysis_summary(experiments, summaries, output_path):
	best_row = summaries.loc[summaries["best_map50_95"].idxmax()]
	lines = [
		"# Experiment Analysis Summary",
		"",
		"## Scope",
		"",
		"This report compares Ultralytics training logs with a consistent metric scope.",
		"",
		"## Results",
		"",
		"| Experiment | Final P | Final R | Final mAP50 | Final mAP50-95 | Best epoch | Best mAP50-95 | Duration (min) |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]

	for _, row in summaries.iterrows():
		duration_minutes = row["duration_seconds"] / 60
		lines.append(
			"| {name} | {precision} | {recall} | {map50} | {map50_95} | "
			"{best_epoch} | {best_value} | {duration} |".format(
				name=row["name"],
				precision=format_number(row["final_precision"]),
				recall=format_number(row["final_recall"]),
				map50=format_number(row["final_map50"]),
				map50_95=format_number(row["final_map50_95"]),
				best_epoch=int(row["best_epoch"]),
				best_value=format_number(row["best_map50_95"]),
				duration=format_number(duration_minutes, 2),
			)
		)

	lines.extend(
		[
			"",
			"## Main Finding",
			"",
			f"The highest best mAP@0.5:0.95 is produced by `{best_row['name']}` "
			f"at epoch {int(best_row['best_epoch'])}: "
			f"{format_number(best_row['best_map50_95'])}.",
			"",
			"## Data Quality Warnings",
			"",
		]
	)

	all_warnings = [
		warning
		for experiment in experiments
		for warning in experiment["warnings"]
	]
	if all_warnings:
		lines.extend(f"- {warning}" for warning in all_warnings)
	else:
		lines.append("- No non-finite values or duplicate epochs were detected.")

	lines.extend(
		[
			"",
			"## Interpretation Rules",
			"",
			"- Use the same dataset split and validation settings for direct comparison.",
			"- Report both final metrics and the best mAP@0.5:0.95 epoch.",
			"- Treat small single-run differences as observations, not universal effects.",
			"- Re-evaluate all final weights on one common test set before ranking models.",
		]
	)

	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(experiment_specs, output_dir, smooth_window=5):
	if smooth_window < 1:
		raise ValueError("smooth-window must be at least 1.")

	output_dir.mkdir(parents=True, exist_ok=True)
	experiments = [build_experiment_data(spec) for spec in experiment_specs]
	summaries = write_summary_tables(experiments, output_dir)
	write_analysis_summary(
		experiments,
		summaries,
		output_dir / "analysis_summary.md",
	)
	plot_metric_curves(
		experiments,
		output_dir / "metrics_over_epochs.png",
		smooth_window,
	)
	plot_loss_curves(
		experiments,
		output_dir / "losses_over_epochs.png",
		smooth_window,
	)
	plot_learning_rate_curves(
		experiments,
		output_dir / "learning_rate_over_epochs.png",
	)
	plot_final_metric_bars(
		summaries,
		output_dir / "final_metrics_comparison.png",
	)
	return summaries


def main():
	arguments = parse_arguments()
	experiment_specs = collect_experiment_specs(
		arguments.input,
		arguments.manifest,
	)
	summaries = run_analysis(
		experiment_specs,
		arguments.output,
		arguments.smooth_window,
	)
	print(summaries.to_string(index=False))
	print(f"Analysis saved to: {arguments.output}")


if __name__ == "__main__":
	main()

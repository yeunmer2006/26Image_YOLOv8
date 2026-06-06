# Experiment Analysis Summary

## Scope

This report compares Ultralytics training logs with a consistent metric scope.

## Results

| Experiment | Final P | Final R | Final mAP50 | Final mAP50-95 | Best epoch | Best mAP50-95 | Duration (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| finetune1 | 0.90961 | 0.83550 | 0.88337 | 0.53988 | 75 | 0.54243 | 97.59 |
| finetune2 | 0.91051 | 0.85209 | 0.89545 | 0.55846 | 85 | 0.55951 | 120.78 |
| finetune3 | 0.90861 | 0.83649 | 0.88425 | 0.54186 | 90 | 0.54324 | 125.11 |

## Main Finding

The highest best mAP@0.5:0.95 is produced by `finetune2` at epoch 85: 0.55951.

## Data Quality Warnings

- finetune1: val/cls_loss contains 100 non-finite values.
- finetune2: val/cls_loss contains 100 non-finite values.

## Interpretation Rules

- Use the same dataset split and validation settings for direct comparison.
- Report both final metrics and the best mAP@0.5:0.95 epoch.
- Treat small single-run differences as observations, not universal effects.
- Re-evaluate all final weights on one common test set before ranking models.

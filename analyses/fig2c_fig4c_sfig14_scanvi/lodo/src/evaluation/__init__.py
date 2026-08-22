"""Evaluation metrics and analysis"""

from .annotation_depth import (
    attach_annotated_depth,
    compute_annotated_depth_series,
    merge_predictions_into_obs,
    plot_annotation_depth_comparison_multi_method,
    save_annotation_depth_benchmark_outputs,
    write_annotation_depth_multi_method_figure,
)
from .metrics import BenchmarkEvaluator, calculate_perclass_metrics

__all__ = [
    "BenchmarkEvaluator",
    "calculate_perclass_metrics",
    "attach_annotated_depth",
    "compute_annotated_depth_series",
    "merge_predictions_into_obs",
    "plot_annotation_depth_comparison_multi_method",
    "save_annotation_depth_benchmark_outputs",
    "write_annotation_depth_multi_method_figure",
]



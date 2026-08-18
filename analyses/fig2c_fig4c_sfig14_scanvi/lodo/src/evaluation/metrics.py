"""
Evaluation metrics for reference mapping benchmarks.

Extracted from baseline_pilot_myeloid/notebooks/05_advanced_evaluation.ipynb
Refactored for multi-lineage reusability.

Provides:
- Per-class metrics (precision, recall, F1, accuracy)
- Confusion matrices
- Summary statistics
- Comparison plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, accuracy_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_label_for_taxonomy(label: str) -> str:
    """
    Normalize cell type labels to match GCA taxonomy.
    
    Handles common naming inconsistencies:
    - "vascular endothelial" → "Endothelial"
    """
    label = str(label)
    # Fix vascular endothelial prefix
    if label.startswith("vascular endothelial"):
        label = label.replace("vascular endothelial", "Endothelial")
    return label


def clean_method_name(method: str) -> str:
    """
    Convert method names for display.
    
    scanvi → SCANVI
    scimilarity_transfer → scimilarity TRANSFER
    scimilarity_zeroshot → scimilarity ZERO-SHOT
    scimilarity_base → scimilarity TRANSFER (legacy)
    scimilarity_lora → scimilarity LoRA
    """
    if method == 'scanvi':
        return 'SCANVI'
    elif method == 'scimilarity_transfer' or method == 'scimilarity_base':
        return 'scimilarity TRANSFER'
    elif method == 'scimilarity_zeroshot':
        return 'scimilarity ZERO-SHOT'
    elif method == 'scimilarity_lora':
        return 'scimilarity LoRA'
    else:
        return method.upper()


def calculate_perclass_metrics(predictions_df: pd.DataFrame, 
                               method_name: str,
                               label_type: str) -> pd.DataFrame:
    """
    Calculate per-class metrics for a single method.
    
    Extracted from baseline notebook 05, Cell 3.
    
    Args:
        predictions_df: DataFrame with columns [cell_id, true_label, predicted_label]
        method_name: Name of the method (e.g., "SCANVI", "scimilarity TRANSFER", "scimilarity ZERO-SHOT")
        label_type: Label type (e.g., ``hgca_celltype_v1``, ``hgca_celltype_level_1``)
        
    Returns:
        DataFrame with per-class precision, recall, F1, accuracy, support
    """
    # Normalize labels to match GCA taxonomy
    predictions_df = predictions_df.copy()
    predictions_df['true_label'] = predictions_df['true_label'].apply(normalize_label_for_taxonomy)
    predictions_df['predicted_label'] = predictions_df['predicted_label'].apply(normalize_label_for_taxonomy)
    
    true_labels = predictions_df['true_label'].values
    pred_labels = predictions_df['predicted_label'].values
    
    # Get unique cell types
    all_labels = sorted(set(true_labels) | set(pred_labels))
    
    # Calculate per-class metrics
    metrics = []
    for cell_type in all_labels:
        # Skip QC labels
        if cell_type in ['doublet', 'lowQ', 'unclear', 'unknown']:
            continue
            
        # Binary classification for this cell type
        y_true = (true_labels == cell_type).astype(int)
        y_pred = (pred_labels == cell_type).astype(int)
        
        # Skip if no instances
        if y_true.sum() == 0:
            continue
        
        # Calculate metrics
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        support = y_true.sum()
        
        metrics.append({
            'method': method_name,
            'label_type': label_type,
            'cell_type': cell_type,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': support
        })
    
    return pd.DataFrame(metrics)


class BenchmarkEvaluator:
    """
    Comprehensive evaluation for reference mapping benchmarks.
    
    Loads predictions from multiple methods and generates:
    - Per-class metrics
    - Confusion matrices
    - Comparison plots
    - Summary statistics
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize evaluator with results directory.
        
        Args:
            results_dir: Path to directory with prediction CSVs
        """
        # Resolve results_dir to an absolute path relative to the repo root
        self.results_dir = Path(results_dir)
        if not self.results_dir.is_absolute():
            module_root = Path(__file__).resolve().parents[2]
            self.results_dir = (module_root / self.results_dir).resolve()

        self.predictions_dir = (self.results_dir / "predictions").resolve()
        self.plots_dir = (self.results_dir / "plots").resolve()
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        (self.plots_dir / "advanced").mkdir(exist_ok=True)
        
        logger.info(f"Initialized BenchmarkEvaluator")
        logger.info(f"  Results: {self.results_dir}")
        logger.info(f"  Predictions: {self.predictions_dir}")
        logger.info(f"  Plots: {self.plots_dir}")
    
    def load_all_predictions(self, 
                           methods: List[str] = None,
                           label_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load prediction CSVs by scanning the predictions directory.
        Ignores relative/CWD issues by using absolute self.predictions_dir.
        
        Filename convention: predictions_{method}_{labeltype}.csv
        Examples:
          predictions_scanvi_hgca_celltype_v1.csv
          predictions_scimilarity_transfer_AI_suggestion.csv
        """
        predictions: Dict[str, pd.DataFrame] = {}

        if not self.predictions_dir.exists():
            logger.warning(f"⚠️  Predictions directory does not exist: {self.predictions_dir}")
            logger.info(f"\n📊 Loaded 0 prediction sets")
            return predictions

        files = sorted(self.predictions_dir.glob("predictions_*.csv"))
        if len(files) == 0:
            logger.warning(f"⚠️  No prediction files found in {self.predictions_dir}")
            logger.info(f"\n📊 Loaded 0 prediction sets")
            return predictions

        for pf in files:
            stem = pf.stem  # predictions_scanvi_hgca_celltype_v1 / predictions_scanvi_level_3_annot
            rest = stem[len("predictions_") :]
            if rest.startswith("scanvi_"):
                method = "scanvi"
                label_raw = rest[len("scanvi_") :]
                key = f"scanvi_{label_raw}"
                label_type = label_raw.replace("_", " ")
            elif rest.startswith("scimilarity_"):
                # e.g. scimilarity_transfer_hgca_celltype_v1
                sub = rest[len("scimilarity_") :]
                sub_parts = sub.split("_", 1)
                if len(sub_parts) < 2:
                    logger.warning(f"Skipping unexpected filename: {pf.name}")
                    continue
                method = f"scimilarity_{sub_parts[0]}"
                label_raw = sub_parts[1]
                key = f"{method}_{label_raw}"
                label_type = label_raw.replace("_", " ")
            else:
                logger.warning(f"Skipping unexpected filename: {pf.name}")
                continue
            try:
                df = pd.read_csv(pf)
                predictions[key] = df
                logger.info(f"✅ Loaded {key}: {len(df):,} predictions from {pf}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {pf}: {e}")

        logger.info(f"\n📊 Loaded {len(predictions)} prediction sets")
        return predictions
    
    def calculate_all_metrics(self, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate per-class metrics for all methods.
        
        Args:
            predictions: Dictionary of prediction DataFrames
            
        Returns:
            Combined DataFrame with all metrics
        """
        logger.info("Calculating per-class metrics for all methods...")
        
        all_metrics = []
        
        for pred_name, pred_df in predictions.items():
            if pred_name.startswith("scanvi_"):
                method_raw = "scanvi"
                label_type = pred_name[len("scanvi_") :].replace("_", " ")
            elif pred_name.startswith("scimilarity_"):
                rest = pred_name[len("scimilarity_") :]
                sub_parts = rest.split("_", 1)
                if len(sub_parts) < 2:
                    continue
                method_raw = f"scimilarity_{sub_parts[0]}"
                label_type = sub_parts[1].replace("_", " ")
            else:
                continue
            
            # Clean method name for display
            method = clean_method_name(method_raw)
            
            logger.info(f"  Calculating for {method} - {label_type}...")
            metrics_df = calculate_perclass_metrics(pred_df, method, label_type)
            all_metrics.append(metrics_df)
        
        # Combine all metrics
        metrics_combined = pd.concat(all_metrics, ignore_index=True)
        
        logger.info(f"✅ Calculated metrics for {len(predictions)} method × label type combinations")
        logger.info(f"📊 Total cell types analyzed: {metrics_combined['cell_type'].nunique()}")
        
        return metrics_combined
    
    def create_f1_comparison_plot(self, 
                                  metrics: pd.DataFrame,
                                  label_type: str,
                                  save: bool = True) -> plt.Figure:
        """
        Create per-class F1 comparison plot.
        
        Args:
            metrics: Metrics DataFrame
            label_type: Which label type to plot
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        metrics_subset = metrics[metrics['label_type'] == label_type].copy()
        
        # Pivot to get methods as columns
        f1_pivot = metrics_subset.pivot(index='cell_type', columns='method', values='f1')
        
        # Sort by average F1
        f1_pivot['avg'] = f1_pivot.mean(axis=1)
        f1_pivot = f1_pivot.sort_values('avg', ascending=True).drop('avg', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(f1_pivot) * 0.4)))
        f1_pivot.plot(kind='barh', ax=ax, width=0.8)
        
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_ylabel('Cell Type', fontsize=12)
        ax.set_title(f'Per-Class F1 Score Comparison\n{label_type} Labels', 
                     fontsize=14, fontweight='bold')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"perclass_f1_comparison_{label_type.replace(' ', '_')}.png"
            plt.savefig(self.plots_dir / "advanced" / filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {filename}")
        
        return fig
    
    def create_confusion_matrix(self,
                               predictions_df: pd.DataFrame,
                               method_name: str,
                               label_type: str,
                               save: bool = True) -> plt.Figure:
        """
        Create confusion matrix plot.
        
        Args:
            predictions_df: Predictions DataFrame
            method_name: Method name for title
            label_type: Label type for title
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        true_labels = predictions_df['true_label'].values
        pred_labels = predictions_df['predicted_label'].values
        
        # Get unique labels (exclude QC)
        all_labels = sorted(set(true_labels) | set(pred_labels))
        valid_labels = [l for l in all_labels if l not in ['doublet', 'lowQ', 'unclear', 'unknown']]
        
        # Filter
        mask = np.isin(true_labels, valid_labels) & np.isin(pred_labels, valid_labels)
        true_filtered = true_labels[mask]
        pred_filtered = pred_labels[mask]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_filtered, pred_filtered, labels=valid_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(10, len(valid_labels) * 0.6), 
                                        max(8, len(valid_labels) * 0.5)))
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Proportion of True Labels', rotation=270, labelpad=20)
        
        # Ticks
        ax.set_xticks(np.arange(len(valid_labels)))
        ax.set_yticks(np.arange(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(valid_labels, fontsize=9)
        
        # Labels
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'Confusion Matrix: {method_name}\n{label_type} Labels', 
                     fontsize=13, fontweight='bold', pad=15)
        
        # Annotations (diagonal and high values)
        for i in range(len(valid_labels)):
            for j in range(len(valid_labels)):
                if i == j or cm_normalized[i, j] > 0.1:
                    text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                           ha="center", va="center", color=text_color, fontsize=7)
        
        plt.tight_layout()
        
        if save:
            filename = f"confusion_matrix_{method_name.replace(' ', '_')}_{label_type.replace(' ', '_')}.png"
            plt.savefig(self.plots_dir / "advanced" / filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {filename}")
        
        return fig
    
    def create_accuracy_heatmap(self,
                               metrics: pd.DataFrame,
                               label_type: str,
                               save: bool = True) -> plt.Figure:
        """
        Create per-class accuracy heatmap.
        
        Args:
            metrics: Metrics DataFrame
            label_type: Which label type to plot
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        metrics_subset = metrics[metrics['label_type'] == label_type].copy()
        
        # Pivot
        acc_pivot = metrics_subset.pivot(index='cell_type', columns='method', values='accuracy')
        
        # Sort by average
        acc_pivot['avg'] = acc_pivot.mean(axis=1)
        acc_pivot = acc_pivot.sort_values('avg', ascending=False).drop('avg', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(8, len(acc_pivot) * 0.35)))
        
        sns.heatmap(acc_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title(f'Per-Class Accuracy Heatmap\n{label_type} Labels',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Cell Type', fontsize=12)
        
        # Rotate x-axis labels to avoid overlap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            filename = f"perclass_accuracy_heatmap_{label_type.replace(' ', '_')}.png"
            plt.savefig(self.plots_dir / "advanced" / filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {filename}")
        
        return fig
    
    def generate_summary_statistics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics across all methods.
        
        Args:
            metrics: Combined metrics DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        logger.info("Generating summary statistics...")
        
        summary = metrics.groupby(['method', 'label_type']).agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'support': 'sum'
        }).round(3)
        
        logger.info("✅ Summary statistics generated")
        
        return summary
    
    def calculate_weighted_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted average metrics (by support).
        
        Args:
            metrics: Metrics DataFrame with support column
            
        Returns:
            DataFrame with weighted averages per method
        """
        weighted_metrics = []
        
        for (method, label_type), group in metrics.groupby(['method', 'label_type']):
            total_support = group['support'].sum()
            
            weighted_f1 = (group['f1'] * group['support']).sum() / total_support
            weighted_acc = (group['accuracy'] * group['support']).sum() / total_support
            weighted_precision = (group['precision'] * group['support']).sum() / total_support
            weighted_recall = (group['recall'] * group['support']).sum() / total_support
            
            weighted_metrics.append({
                'method': method,
                'label_type': label_type,
                'weighted_f1': weighted_f1,
                'weighted_accuracy': weighted_acc,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'total_support': total_support
            })
        
        return pd.DataFrame(weighted_metrics)
    
    def create_usage_heatmap(self, predictions: Dict[str, pd.DataFrame], 
                            label_type: str, 
                            save: bool = True) -> plt.Figure:
        """
        Create a heatmap showing how many cells of each true type were predicted as each type.
        This helps diagnose label matching and coverage issues.
        
        Args:
            predictions: Dictionary of prediction DataFrames keyed by "{method}_{label_type}"
            label_type: Which label type to plot (e.g., ``hgca_celltype_v1``)
            save: Whether to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating usage heatmap for {label_type}...")
        
        # Combine all predictions across methods
        all_preds = []
        for key, df in predictions.items():
            if label_type.replace(' ', '_') in key:
                method = key.split('_')[0]
                if method == 'scimilarity':
                    method = f"scimilarity_{key.split('_')[1]}"
                df_copy = df.copy()
                df_copy['method'] = clean_method_name(method)
                # Apply label normalization
                df_copy['true_label'] = df_copy['true_label'].apply(normalize_label_for_taxonomy)
                df_copy['predicted_label'] = df_copy['predicted_label'].apply(normalize_label_for_taxonomy)
                all_preds.append(df_copy)
        
        if not all_preds:
            logger.warning(f"No predictions found for {label_type}")
            return None
        
        combined = pd.concat(all_preds, ignore_index=True)
        
        # Count predictions per (true_label, predicted_label) pair
        usage = combined.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
        
        # Get top cell types by frequency
        top_true = usage.groupby('true_label')['count'].sum().nlargest(30).index
        top_pred = usage.groupby('predicted_label')['count'].sum().nlargest(30).index
        
        # Filter to top types
        usage_filt = usage[usage['true_label'].isin(top_true) & usage['predicted_label'].isin(top_pred)]
        
        # Pivot for heatmap
        usage_pivot = usage_filt.pivot_table(
            index='true_label', 
            columns='predicted_label', 
            values='count', 
            fill_value=0
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        
        sns.heatmap(usage_pivot, annot=False, cmap='YlOrRd', 
                   cbar_kws={'label': 'Cell Count'},
                   linewidths=0.5, linecolor='lightgray', ax=ax,
                   fmt='g')
        
        ax.set_title(f'Cell Type Usage Matrix: True vs Predicted\n{label_type} (Top 30 types, all methods combined)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Cell Type', fontsize=12)
        ax.set_ylabel('True Cell Type', fontsize=12)
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filename = f"usage_heatmap_{label_type.replace(' ', '_')}.png"
            plt.savefig(self.plots_dir / "advanced" / filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {filename}")
        
        return fig
    
    def run_full_evaluation(self, 
                           methods: List[str] = None,
                           label_types: List[str] = None,
                           create_plots: bool = True) -> Dict:
        """
        Run complete evaluation pipeline.
        
        This is the main method that orchestrates everything:
        1. Load all predictions
        2. Calculate metrics
        3. Generate plots
        4. Save results
        
        Args:
            methods: Methods to evaluate
            label_types: Label types to evaluate
            create_plots: Whether to create and save plots
            
        Returns:
            Dictionary with all results and metrics
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING FULL EVALUATION PIPELINE")
        logger.info("="*70 + "\n")
        
        # Load predictions
        predictions = self.load_all_predictions(methods, label_types)
        
        if len(predictions) == 0:
            raise ValueError(f"No prediction files found at {self.predictions_dir} for methods={methods} label_types={label_types}")
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(predictions)
        
        # Save metrics
        metrics_file = self.results_dir / "perclass_metrics.csv"
        metrics.to_csv(metrics_file, index=False)
        logger.info(f"✅ Saved per-class metrics: {metrics_file.name}")
        
        # Calculate weighted metrics
        weighted = self.calculate_weighted_metrics(metrics)
        weighted_file = self.results_dir / "weighted_metrics.csv"
        weighted.to_csv(weighted_file, index=False)
        logger.info(f"✅ Saved weighted metrics: {weighted_file.name}")
        
        # Generate summary
        summary = self.generate_summary_statistics(metrics)
        summary_file = self.results_dir / "summary_statistics.csv"
        summary.to_csv(summary_file)
        logger.info(f"✅ Saved summary statistics: {summary_file.name}")
        
        # Create plots if requested
        if create_plots:
            logger.info("\n📊 Creating evaluation plots...")
            
            for label_type in metrics['label_type'].unique():
                # F1 comparison
                self.create_f1_comparison_plot(metrics, label_type)
                
                # Accuracy heatmap
                self.create_accuracy_heatmap(metrics, label_type)
                
                # Confusion matrices for each method
                for pred_name, pred_df in predictions.items():
                    if label_type.replace(' ', '_') in pred_name:
                        parts = pred_name.split('_')
                        if parts[0] == 'scanvi':
                            method = 'SCANVI'
                        elif parts[0] == 'scimilarity':
                            method = f"scimilarity {parts[1]}"
                        else:
                            continue
                        
                        self.create_confusion_matrix(pred_df, method, label_type)
        
        # Return everything
        return {
            'predictions': predictions,
            'perclass_metrics': metrics,
            'weighted_metrics': weighted,
            'summary_statistics': summary
        }


# Example usage
if __name__ == "__main__":
    print("Benchmark Evaluation Module")
    print("\nKey Functions:")
    print("  - calculate_perclass_metrics(): Per-class precision/recall/F1")
    print("  - BenchmarkEvaluator: Complete evaluation pipeline")
    print("\nExample:")
    print("""
    from src.evaluation.metrics import BenchmarkEvaluator
    
    evaluator = BenchmarkEvaluator('results/myeloid')
    results = evaluator.run_full_evaluation()
    
    # Access metrics
    print(results['weighted_metrics'])
    print(results['summary_statistics'])
    """)


"""
SCANVI benchmark implementation.

Extracted from baseline_pilot_myeloid/notebooks/02_scanvi_benchmark.ipynb
Refactored for multi-lineage use and disease projection.

Key Design:
- SCANVITrainer: Train models on reference data
- SCANVIPredictor: Apply trained models to new data (disease datasets!)
- Supports scVI pretraining for better embeddings (recommended workflow)
"""

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time
import logging

# scVI/SCANVI imports
from scvi.model import SCVI, SCANVI
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.evaluation.annotation_depth import (
    merge_predictions_into_obs,
    save_annotation_depth_benchmark_outputs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_batch_size_for_scvi_train_splits(
    requested_bs: int,
    n_obs: int,
    split_train_sizes: list[int],
) -> int:
    """
    Choose batch size so no training split yields a final mini-batch of size 1.

    PyTorch BatchNorm in training mode errors on batch size 1; scvi/Lightning can
    emit a remainder-1 last batch when ``n_train % batch_size == 1`` (unless
    ``drop_last=True`` on the dataloader, which this module sets by default).
    """
    bs = min(int(requested_bs), max(1, n_obs))
    for _ in range(max(bs, 2048)):
        if bs < 1:
            return 1
        bad = False
        for n in split_train_sizes:
            if n < 2:
                continue
            if bs >= 2 and n > bs and n % bs == 1:
                bad = True
                break
        if not bad:
            return max(1, bs)
        bs -= 1
    return max(1, bs)


def _datasplitter_kwargs_for_train(config: Dict) -> Dict[str, Any]:
    """
    scvi / Lightning dataloaders use ``drop_last=False`` by default, which can leave
    a final batch of 1 and trigger BatchNorm training errors. Pass ``drop_last=True``;
    overridable via config ``datasplitter_kwargs`` (e.g. for compatibility tests).
    """
    out = dict(config.get("datasplitter_kwargs") or {})
    out.setdefault("drop_last", True)
    return out


class FrozenLatentModel:
    """
    Precomputed latent (e.g. ``obsm['X_scANVI']``) for mapQC, which expects
    ``get_latent_representation()`` like scVI/SCANVI.
    """

    def __init__(
        self,
        adata_train: anndata.AnnData,
        adata_test: anndata.AnnData,
        key: str = "X_scanvi",
    ):
        self._adata_train = adata_train
        self._adata_test = adata_test
        self.key = key

    def get_latent_representation(self, adata: Optional[anndata.AnnData] = None):
        if adata is None:
            return np.asarray(self._adata_train.obsm[self.key])
        ta = np.asarray(self._adata_train.obsm[self.key])
        te = np.asarray(self._adata_test.obsm[self.key])
        # mapQC uses copies of train/test — match by obs names, not object identity
        if list(adata.obs_names) == list(self._adata_train.obs_names):
            return ta
        if list(adata.obs_names) == list(self._adata_test.obs_names):
            return te
        raise ValueError(
            "FrozenLatentModel.get_latent_representation: adata rows must match train or test obs_names."
        )


class SCANVITrainer:
    """
    Train SCANVI models for cell type annotation.
    
    Designed for reference mapping benchmark - trains on reference data
    and evaluates on held-out test set.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SCANVI trainer with configuration.
        
        Args:
            config: Dictionary with SCANVI hyperparameters
                Required keys: n_latent, n_layers, dropout_rate, gene_likelihood,
                               n_epochs, batch_size, learning_rate
                Optional: use_scvi_pretrain (bool), scvi_epochs (int)
        """
        self.config = config
        self.use_scvi_pretrain = config.get('use_scvi_pretrain', False)
        logger.info(f"Initialized SCANVI trainer (scVI pretrain: {self.use_scvi_pretrain})")
    
    def train_and_evaluate(self, 
                          adata_train: anndata.AnnData,
                          adata_test: anndata.AnnData,
                          label_key: str,
                          batch_key: Optional[str] = None,
                          *,
                          split_meta: Optional[Dict[str, Any]] = None,
                          annotation_depth_output_dir: Optional[str] = None,
                          ) -> Dict:
        """
        Train SCANVI and evaluate on test set.
        
        This is the reference mapping benchmark approach:
        1. Train on adata_train
        2. Predict on adata_test (held-out)
        3. Calculate accuracy metrics
        4. Save predictions
        
        Args:
            adata_train: Training data (preprocessed)
            adata_test: Test data (preprocessed)
            label_key: Column name for cell type labels
            batch_key: Optional batch correction key
            
        Returns:
            Dictionary with results, predictions, and metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"SCANVI Training and Evaluation: {label_key}")
        logger.info(f"{'='*70}")
        
        logger.info(f"Train: {adata_train.n_obs:,} cells")
        logger.info(f"Test:  {adata_test.n_obs:,} cells")
        logger.info(f"Labels: {adata_train.obs[label_key].nunique()} types")

        if self.config.get("use_pretrained_embedding"):
            return self._train_and_evaluate_frozen_latent(
                adata_train,
                adata_test,
                label_key,
                batch_key=batch_key,
                split_meta=split_meta,
                annotation_depth_output_dir=annotation_depth_output_dir,
            )
        
        # Setup SCANVI
        logger.info("Setting up SCANVI model...")
        
        # Set batch key - must be done BEFORE setup_anndata
        if batch_key is None:
            batch_key = "batch"
        
        if batch_key not in adata_train.obs.columns:
            # Create dummy batch if not present
            logger.info(f"Creating dummy batch column: {batch_key}")
            adata_train.obs[batch_key] = "batch_1"
            adata_test.obs[batch_key] = "batch_1"
        else:
            # Batch key exists - make categories consistent
            # Convert to categorical with all categories from both train and test
            all_batches = pd.concat([adata_train.obs[batch_key], adata_test.obs[batch_key]]).unique()
            logger.info(f"Using batch key: {batch_key} ({len(all_batches)} batches)")
            adata_train.obs[batch_key] = pd.Categorical(
                adata_train.obs[batch_key], 
                categories=all_batches
            )
            adata_test.obs[batch_key] = pd.Categorical(
                adata_test.obs[batch_key],
                categories=all_batches
            )

        train_size = float(self.config.get("train_size", 0.9))
        req_bs = int(self.config.get("batch_size", 256))
        uc = str(self.config.get("unlabeled_category", "Unknown"))
        n = adata_train.n_obs
        n_tr = max(1, int(n * train_size))
        n_lab = int((adata_train.obs[label_key].astype(str) != uc).sum())
        n_lb_tr = max(1, int(n_lab * train_size))

        if self.use_scvi_pretrain:
            bs_scvi = _safe_batch_size_for_scvi_train_splits(req_bs, n, [n_tr])
            bs_scanvi = _safe_batch_size_for_scvi_train_splits(req_bs, n, [n_tr, n_lb_tr])
            batch_size = min(bs_scvi, bs_scanvi)
        else:
            batch_size = _safe_batch_size_for_scvi_train_splits(req_bs, n, [n_tr, n_lb_tr])

        if batch_size != req_bs:
            logger.info(
                f"Adjusted batch_size {req_bs} -> {batch_size} "
                f"(avoid remainder-1 last batch; n_train≈{n_tr}, "
                f"n_labeled_train≈{n_lb_tr})"
            )

        if n_tr < 2:
            raise ValueError(
                f"Too few training cells after scVI train/val split (~{train_size=}, "
                f"n_obs={n} → n_train≈{n_tr}). scVI/SCANVI + BatchNorm need at least 2."
            )

        dsk = _datasplitter_kwargs_for_train(self.config)

        def _train_with_drop_last(m, **kwargs: Any) -> None:
            try:
                m.train(**kwargs, datasplitter_kwargs=dsk)
            except TypeError as err:
                if "datasplitter_kwargs" not in str(err):
                    raise
                logger.warning(
                    "scvi train() rejected datasplitter_kwargs=%s (%s); retrying without it",
                    dsk,
                    err,
                )
                m.train(**kwargs)

        # Initialize SCANVI model (with optional scVI pretraining)
        start_time = time.time()
        
        if self.use_scvi_pretrain:
            # RECOMMENDED WORKFLOW: Train scVI first, then SCANVI
            logger.info("=" * 70)
            logger.info("STEP 1: Training scVI (unsupervised pretraining)")
            logger.info("=" * 70)
            
            # Setup for scVI (unsupervised)
            SCVI.setup_anndata(
                adata_train,
                batch_key=batch_key,
                layer='counts'
            )
            
            # Initialize scVI
            scvi_model = SCVI(
                adata_train,
                n_latent=self.config.get('n_latent', 30),
                n_layers=self.config.get('n_layers', 2),
                dropout_rate=self.config.get('dropout_rate', 0.1),
                gene_likelihood=self.config.get('gene_likelihood', 'nb')
            )
            
            # Train scVI (unsupervised)
            scvi_epochs = self.config.get('scvi_epochs', 10)
            logger.info(f"Training scVI for {scvi_epochs} epochs...")
            scvi_start = time.time()
            _train_with_drop_last(
                scvi_model,
                max_epochs=scvi_epochs,
                train_size=train_size,
                batch_size=batch_size,
            )
            scvi_time = time.time() - scvi_start
            logger.info(f"✅ scVI trained in {scvi_time:.1f}s")
            
            # Initialize SCANVI from scVI
            logger.info("=" * 70)
            logger.info("STEP 2: Initializing SCANVI from scVI")
            logger.info("=" * 70)
            model = SCANVI.from_scvi_model(
                scvi_model,
                labels_key=label_key,
                unlabeled_category=self.config.get('unlabeled_category', 'Unknown')
            )
            logger.info("✅ SCANVI initialized from scVI (warm start)")
            
        else:
            # STANDARD WORKFLOW: Train SCANVI from scratch
            logger.info("Training SCANVI from scratch (no scVI pretraining)")
            
            # Setup anndata for SCANVI
            SCANVI.setup_anndata(
                adata_train,
                labels_key=label_key,
                batch_key=batch_key,
                layer='counts',
                unlabeled_category=self.config.get('unlabeled_category', 'Unknown')
            )
            
            # Initialize SCANVI
            model = SCANVI(
                adata_train,
                n_latent=self.config.get('n_latent', 30),
                n_layers=self.config.get('n_layers', 2),
                dropout_rate=self.config.get('dropout_rate', 0.1),
                gene_likelihood=self.config.get('gene_likelihood', 'nb')
            )
            logger.info("✅ SCANVI model initialized")
        
        # Train
        logger.info("🚀 Training SCANVI...")
        logger.info(f"   Epochs: {self.config.get('n_epochs', 10)}")
        logger.info(f"   Batch size: {batch_size}")

        _train_with_drop_last(
            model,
            max_epochs=self.config.get("n_epochs", 10),
            train_size=train_size,
            early_stopping=self.config.get("early_stopping", False),
            early_stopping_patience=self.config.get("early_stopping_patience", 20),
            batch_size=batch_size,
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Predict on test set
        logger.info("🔮 Predicting on test set...")
        # Batch categories already aligned above, so predict should work
        predictions = model.predict(adata_test)
        true_labels = adata_test.obs[label_key].values
        
        # Calculate accuracy
        accuracy = (predictions == true_labels).mean()
        logger.info(f"✅ Accuracy: {accuracy:.3f}")
        
        # Prepare results
        results = {
            'method': 'SCANVI',
            'label_type': label_key,
            'n_train': adata_train.n_obs,
            'n_test': adata_test.n_obs,
            'n_genes': adata_train.n_vars,
            'n_labels': adata_train.obs[label_key].nunique(),
            'training_time': training_time,
            'accuracy': float(accuracy),
            'model': model  # Keep model for saving
        }
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'cell_id': adata_test.obs_names.values,
            'true_label': true_labels,
            'predicted_label': predictions
        })
        
        results['predictions'] = predictions_df

        export_dir = annotation_depth_output_dir or self.config.get("annotation_depth_output_dir")
        if export_dir:
            ad_annot = merge_predictions_into_obs(adata_test, predictions, label_key)
            save_annotation_depth_benchmark_outputs(
                ad_annot,
                export_dir,
                results["method"],
                label_key=label_key,
                split_meta=split_meta,
            )
        
        logger.info(f"✅ SCANVI benchmark complete for {label_key}")
        
        return results

    def _train_and_evaluate_frozen_latent(
        self,
        adata_train: anndata.AnnData,
        adata_test: anndata.AnnData,
        label_key: str,
        batch_key: Optional[str] = None,
        *,
        split_meta: Optional[Dict[str, Any]] = None,
        annotation_depth_output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Multinomial logistic regression on ``obsm['X_scanvi']`` (from precomputed scANVI/SCANVI).
        """
        if "X_scanvi" not in adata_train.obsm or "X_scanvi" not in adata_test.obsm:
            raise KeyError(f"Frozen latent mode requires obsm['X_scanvi'] on train and test (after preprocessing).")

        logger.info("Frozen latent: multinomial logistic regression on X_scanvi (no scVI training)")
        start_time = time.time()

        X_train = np.asarray(adata_train.obsm["X_scanvi"], dtype=np.float32)
        X_test = np.asarray(adata_test.obsm["X_scanvi"], dtype=np.float32)
        y_train_raw = adata_train.obs[label_key].astype(str).values
        y_test_raw = adata_test.obs[label_key].astype(str).values

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        clf = LogisticRegression(
            solver=str(self.config.get("frozen_latent_solver", "saga")),
            max_iter=int(self.config.get("frozen_latent_max_iter", 500)),
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred_idx = clf.predict(X_test)
        predictions = le.inverse_transform(y_pred_idx)

        training_time = time.time() - start_time
        accuracy = float((predictions == y_test_raw).mean())
        logger.info(f"✅ Accuracy (frozen latent): {accuracy:.3f}")

        results = {
            "method": "SCANVI",
            "label_type": label_key,
            "n_train": adata_train.n_obs,
            "n_test": adata_test.n_obs,
            "n_genes": adata_train.n_vars,
            "n_labels": int(len(le.classes_)),
            "training_time": training_time,
            "accuracy": float(accuracy),
            "model": FrozenLatentModel(adata_train, adata_test, "X_scanvi"),
        }

        predictions_df = pd.DataFrame(
            {
                "cell_id": adata_test.obs_names.values,
                "true_label": y_test_raw,
                "predicted_label": predictions,
            }
        )
        results["predictions"] = predictions_df

        export_dir = annotation_depth_output_dir or self.config.get("annotation_depth_output_dir")
        if export_dir:
            ad_annot = merge_predictions_into_obs(adata_test, predictions, label_key)
            save_annotation_depth_benchmark_outputs(
                ad_annot,
                export_dir,
                results["method"],
                label_key=label_key,
                split_meta=split_meta,
            )

        logger.info(f"✅ Frozen latent benchmark complete for {label_key}")
        return results
    
    def save_model(self, model, save_path: str) -> None:
        """
        Save trained SCANVI model.
        
        Args:
            model: Trained SCANVI model
            save_path: Path to save directory
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path, overwrite=True)
        logger.info(f"✅ Saved SCANVI model to {save_path}")


class SCANVIPredictor:
    """
    Apply trained SCANVI model to new data.
    
    CRITICAL for disease projection - loads pre-trained model
    and applies it to unseen data without retraining.
    """
    
    def __init__(self, model_path: str, adata_reference: anndata.AnnData):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved SCANVI model
            adata_reference: Reference data used for training (for gene alignment)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading SCANVI model from {model_path}")
        self.model = SCANVI.load(model_path, adata_reference)
        logger.info(f"✅ Model loaded successfully")
    
    def predict(self, adata_new: anndata.AnnData) -> np.ndarray:
        """
        Predict cell types for new data.
        
        This is used for:
        1. Validation (apply to test set from same distribution)
        2. Disease projection (apply to disease dataset!)
        
        Args:
            adata_new: New data to predict on (preprocessed)
            
        Returns:
            Array of predicted cell type labels
        """
        logger.info(f"Predicting on {adata_new.n_obs:,} cells...")
        
        predictions = self.model.predict(adata_new)
        
        logger.info(f"✅ Predictions complete")
        
        return predictions
    
    def predict_with_uncertainty(self, adata_new: anndata.AnnData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Returns both predictions and probability matrix.
        Useful for threshold-based filtering.
        
        Args:
            adata_new: New data to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info(f"Predicting with probabilities on {adata_new.n_obs:,} cells...")
        
        # Get soft predictions (probability matrix)
        probs = self.model.predict(adata_new, soft=True)
        
        # Get hard predictions
        predictions = self.model.predict(adata_new)
        
        logger.info(f"✅ Predictions with probabilities complete")
        
        return predictions, probs


# Example usage and testing
if __name__ == "__main__":
    print("SCANVI Benchmark Module")
    print("\nClasses:")
    print("  - SCANVITrainer: Train on reference data")
    print("  - SCANVIPredictor: Apply to new data (disease projection!)")
    print("\nExample usage in notebook:")
    print("""
    # Training
    from src.models.scanvi_benchmark import SCANVITrainer
    
    trainer = SCANVITrainer(config['scanvi'])
    results = trainer.train_and_evaluate(adata_train, adata_test, label_key)
    trainer.save_model(results['model'], 'models/scanvi_myeloid.pkl')
    
    # Disease projection (later)
    from src.models.scanvi_benchmark import SCANVIPredictor
    
    predictor = SCANVIPredictor('models/scanvi_myeloid.pkl', adata_reference)
    predictions = predictor.predict(adata_disease)  # No retraining!
    """)


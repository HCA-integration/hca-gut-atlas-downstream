"""
Data preparation for reference mapping benchmarks.

Extracted from notebooks 02 & 03, refactored for reusability.
Key design: Separate preprocessing (applies to ANY dataset) from 
training-specific operations (only for reference data).
"""

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from sklearn.model_selection import train_test_split
import logging

from src.utils.scimilarity_align import align_anndata_to_scimilarity
from src.utils.hgca_obs import ensure_var_index_gene_symbols, normalize_hgca_obs_columns
from src.utils.atlas_lineage_mapping import benchmark_lineage_from_pangi_level1
from src.utils.splits import iter_leave_one_dataset_folds, train_test_mask_leave_one_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Handles data loading and preprocessing for reference mapping.
    
    Design Philosophy:
    - preprocess() works on ANY dataset (including disease data)
    - split_for_training() only called when training models
    - Methods are modular and composable
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data preparation with config.
        
        Args:
            config: Configuration dictionary with data paths and parameters
        """
        self.config = config
        self.lineage = config['lineage']
        self.data_path = Path(config['data']['input_path'])
        
        logger.info(f"Initialized DataPreparation for {self.lineage}")
        # False = vignette-style: avoid adata.copy() + counts duplicate before align (large HGCA objects)
        self._scimilarity_preprocess_copy_adata = bool(
            self.config.get("data", {}).get("scimilarity_preprocessing_copy_adata", False)
        )
        # Train/test split: False uses AnnData row views (no extra X copy); True materializes copies (default)
        self._split_copy_adata = bool(self.config.get("data", {}).get("split_copy_adata", True))

    def _excluded_supervised_label_strings(self, label_key: str) -> set:
        """
        Extra label values to drop for supervised training/benchmarking (e.g. coarse bucket ``Epithelial``
        in ``hgca_celltype_v1``). Config: ``training.exclude_hgca_celltype_v1_labels`` (list of strings).
        Default for ``hgca_celltype_v1``: ``["Epithelial"]``. Set to ``[]`` in YAML to disable.
        """
        training = self.config.get("training") or {}
        if label_key != "hgca_celltype_v1":
            return set()
        raw = training.get("exclude_hgca_celltype_v1_labels")
        if raw is None:
            raw = ["Epithelial"]
        return {str(x) for x in raw}

    def _log_config_label_columns(self, adata: anndata.AnnData) -> None:
        """Log coverage for configured ``labels`` (e.g. ``hgca_celltype_v1``, ``hgca_celltype_level_1``)."""
        for col in self.config.get("labels") or []:
            if col not in adata.obs.columns:
                logger.warning("   Config label %r not in adata.obs", col)
                continue
            s = adata.obs[col]
            n = int(s.notna().sum())
            n_types = int(s.nunique(dropna=True))
            logger.info("   %s: %s cells, %s types", col, f"{n:,}", n_types)
    
    def load_lineage_data(self) -> anndata.AnnData:
        """
        Load lineage data from 10X format.
        
        Returns:
            AnnData object with raw counts
            
        Raises:
            FileNotFoundError: If data files don't exist
        """
        logger.info(f"Loading {self.lineage} data from {self.data_path}")
        
        # Check if path exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        # Determine format and load appropriately
        data_format = self.config['data'].get('format', 'auto')
        
        if data_format == 'h5ad' or str(self.data_path).endswith('.h5ad'):
            backed = bool(self.config.get("data", {}).get("load_backed", False))
            logger.info("Loading h5ad file%s...", " (backed)" if backed else "")
            adata = sc.read_h5ad(self.data_path, backed="r" if backed else None)
            logger.info(f"✅ Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        elif data_format == '10x' or self.data_path.is_dir():
            # Load 10X format
            logger.info(f"Loading 10X format...")
            try:
                adata = sc.read_10x_mtx(
                    self.data_path,
                    var_names='gene_symbols',
                    cache=True
                )
                logger.info(f"✅ Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
                
            except Exception as e:
                logger.error(f"Failed to load 10X data: {e}")
                raise
        else:
            raise ValueError(f"Unknown data format: {data_format} for path {self.data_path}")

        ls = self.config.get("data", {}).get("lineage_subset")
        if ls and isinstance(ls, dict):
            col = ls.get("column")
            val = ls.get("value")
            vals = ls.get("values")
            if col and col in adata.obs.columns:
                if vals is not None:
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]
                    want = [str(v) for v in vals]
                    m = adata.obs[col].astype(str).isin(want)
                    n_sub = int(m.sum())
                    logger.info(
                        "Lineage subset: %s in %s → %s cells",
                        col,
                        want,
                        f"{n_sub:,}",
                    )
                elif val is not None:
                    m = adata.obs[col].astype(str) == str(val)
                    n_sub = int(m.sum())
                    logger.info(
                        "Lineage subset: %s == %r → %s cells",
                        col,
                        val,
                        f"{n_sub:,}",
                    )
                else:
                    raise ValueError(
                        "lineage_subset requires 'value' or 'values' when column is set."
                    )
                adata = adata[m]
                if getattr(adata, "isbacked", False):
                    adata = adata.to_memory()
            else:
                raise KeyError(
                    f"lineage_subset.column={col!r} not found in obs (have lineage subset config)."
                )

        if "level_1_annot" in adata.obs.columns:
            adata.obs["atlas_benchmark_lineage"] = [
                benchmark_lineage_from_pangi_level1(x) or ""
                for x in adata.obs["level_1_annot"].values
            ]
        
        # Load metadata if available (only for 10X format - h5ad already has it)
        if data_format != 'h5ad' and not str(self.data_path).endswith('.h5ad'):
            metadata_path = self.data_path.parent / f"{self.lineage}_metadata.csv"
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path, index_col=0)
                # Merge with obs (careful with index alignment)
                common_cells = adata.obs.index.intersection(metadata.index)
                if len(common_cells) > 0:
                    adata.obs = adata.obs.join(metadata, how='left')
                    logger.info(f"✅ Merged metadata for {len(common_cells):,} cells")
        
        # Optional: legacy cluster CSV (taxonomy.path) — NOT used when integrated HGCA obs has labels
        if "taxonomy" in self.config:
            tax = self.config["taxonomy"]
            if tax.get("path"):
                skip = bool(tax.get("skip_external_annotations", False))
                labels_cfg = self.config.get("labels") or []
                labels_in_obs = labels_cfg and all(
                    c in adata.obs.columns for c in labels_cfg
                )

                if skip:
                    logger.info(
                        "Skipping external annotation CSV (taxonomy.skip_external_annotations=true). "
                        "Use hgca_celltype_* columns already in obs."
                    )
                elif labels_in_obs:
                    logger.info(
                        "Skipping external annotation CSV: all config labels %s are already in obs.",
                        labels_cfg,
                    )
                else:
                    logger.info(
                        "Config labels not fully in obs; loading legacy annotation CSV from taxonomy.path"
                    )
                    adata = self._load_annotations(adata)
            elif tax.get("skip_external_annotations"):
                logger.info(
                    "Skipping external annotation CSV (no taxonomy.path; skip_external_annotations)."
                )

        self._log_config_label_columns(adata)
        return adata
    
    def _load_annotations(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Load and merge cluster annotations from ``taxonomy.path`` (legacy lineage CSV).

        Merges on ``annotation_cluster``. For integrated HGCA objects, prefer
        ``hgca_celltype_v1`` / ``hgca_celltype_level_1`` in ``obs`` and set
        ``taxonomy.skip_external_annotations: true``.
        """
        annot_path = Path(self.config['taxonomy']['path'])
        if not annot_path.exists():
            logger.warning(f"Annotation file not found: {annot_path}")
            return adata
        
        logger.info(f"Loading annotations from {annot_path.name}")
        annotations = pd.read_csv(annot_path)
        
        # Check if we have annotation_cluster column
        if 'annotation_cluster' not in adata.obs.columns:
            logger.warning("No 'annotation_cluster' column found in adata.obs")
            logger.warning("Available columns: " + str(list(adata.obs.columns[:10])))
            return adata
        
        # Merge annotations on annotation_cluster
        # This is the method used in the baseline notebooks
        annotations_renamed = annotations.rename(columns={"Cluster": "annotation_cluster"})
        
        # Merge (left join to keep all cells)
        adata.obs = adata.obs.merge(
            annotations_renamed,
            on='annotation_cluster',
            how='left',
            suffixes=('_orig', '')
        )
        
        logger.info("✅ Merged annotations")

        for col in self.config.get("labels") or []:
            if col not in adata.obs.columns:
                continue
            ser = adata.obs[col]
            if str(ser.dtype) == "category":
                continue
            if ser.dtype == object or str(ser.dtype) == "string":
                adata.obs[col] = ser.astype(str).str.replace("\n", " ", regex=False)
                logger.info("   ✅ Cleaned newlines in %s", col)

        return adata
    
    def preprocess(self, adata: anndata.AnnData, 
                   for_method: str = "general",
                   copy_adata: Optional[bool] = None) -> anndata.AnnData:
        """
        Preprocess data - works on ANY dataset (reference OR disease).
        
        This method does NOT split data or do training-specific operations.
        It can be applied to new/unseen disease datasets.
        
        Args:
            adata: Input AnnData object
            for_method: "scanvi", "scimilarity", or "general"
            copy_adata: If True, copy ``adata`` before changes. If None, use config
                ``data.scimilarity_preprocessing_copy_adata`` for scimilarity (default False),
                and True for scanvi/general.
            
        Returns:
            Preprocessed AnnData object
        """
        logger.info(f"Preprocessing data for {for_method}")
        if copy_adata is None:
            copy_adata = (
                True
                if for_method in ("scanvi", "general")
                else self._scimilarity_preprocess_copy_adata
            )

        sv_cfg = self.config.get("scanvi") or {}
        if for_method in ("scanvi", "general") and sv_cfg.get("use_pretrained_embedding"):
            if copy_adata:
                adata = adata.copy()
            key = str(sv_cfg.get("pretrained_obsm_key", "X_scANVI"))
            if key not in adata.obsm:
                raise KeyError(
                    f"scanvi.use_pretrained_embedding requires obsm[{key!r}] (from CELLxGENE / prior scANVI run)."
                )
            z = np.asarray(adata.obsm[key], dtype=np.float32)
            adata.obsm["X_scanvi"] = z
            logger.info(
                "✅ Frozen embedding mode: using obsm[%r] → X_scanvi (%s × %s), skipping HVG/counts",
                key,
                f"{z.shape[0]:,}",
                z.shape[1],
            )
            return adata

        if copy_adata:
            adata = adata.copy()
        
        # Store raw counts FIRST (before any filtering)
        if 'counts' not in adata.layers:
            if copy_adata:
                adata.layers['counts'] = adata.X.copy()
            else:
                # Same reference as vignettes (scimilarity_in_context): lognorm_counts will copy to X
                adata.layers['counts'] = adata.X
        
        if for_method in ["scanvi", "general"]:
            # SCANVI preprocessing (includes gene filtering and HVG selection)
            logger.info("Filtering genes and cells for SCANVI...")
            sc.pp.filter_genes(adata, min_cells=10)
            adata = self._prepare_for_scanvi(adata)
        elif for_method == "scimilarity":
            # scimilarity preprocessing (NO filtering - keeps ALL genes!)
            logger.info("Preparing for scimilarity (keeping all genes, no filtering)...")
            # Don't filter! scimilarity needs all genes for alignment
            adata = self._prepare_for_scimilarity(adata)
        
        logger.info(f"✅ Preprocessing complete: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        return adata
    
    def _prepare_for_scanvi(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        SCANVI-specific preprocessing.
        
        Extracted from baseline_pilot_myeloid/notebooks/02_scanvi_benchmark.ipynb
        """
        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Highly variable genes
        n_hvg = min(4000, adata.n_vars)
        
        # Try different HVG methods
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True, flavor='seurat_v3')
            logger.info(f"✅ Selected {n_hvg} HVGs (seurat_v3)")
        except:
            logger.warning("seurat_v3 failed, trying seurat")
            try:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True, flavor='seurat')
                logger.info(f"✅ Selected {n_hvg} HVGs (seurat)")
            except:
                logger.warning("seurat failed, trying cell_ranger")
                sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True, flavor='cell_ranger')
                logger.info(f"✅ Selected {n_hvg} HVGs (cell_ranger)")
        
        return adata
    
    def _prepare_for_scimilarity(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        scimilarity-specific preprocessing.
        
        Extracted from baseline_pilot_myeloid/notebooks/03_scimilarity_benchmark.ipynb
        
        IMPORTANT: Do NOT filter to HVGs - scimilarity needs ALL genes!
        """
        # Basic QC only (no HVG filtering!)
        # scimilarity will align to its gene order later
        
        # Ensure var.index contains gene symbols
        if 'gene_symbol' in adata.var.columns:
            adata.var.set_index('gene_symbol', inplace=True, drop=False)
        elif adata.var.index.name != 'gene_symbol':
            # Try to find gene symbols
            if 'gene_ids' in adata.var.columns:
                logger.warning("Using gene_ids as gene symbols")
                adata.var.set_index('gene_ids', inplace=True, drop=False)
        
        # Remove duplicate gene symbols
        if adata.var.index.has_duplicates:
            n_dup = adata.var.index.duplicated().sum()
            logger.warning(f"Removing {n_dup} duplicated gene symbols")
            adata = adata[:, ~adata.var.index.duplicated(keep='first')].copy()
        
        # scimilarity expects ALL genes - will be aligned in model class
        logger.info(f"Prepared for scimilarity: {adata.n_vars:,} unique genes (no HVG filtering)")
        
        return adata
    
    def _filter_valid_label_cells(
        self,
        adata: anndata.AnnData,
        label_key: str,
        copy: bool,
    ) -> Tuple[anndata.AnnData, pd.Series]:
        """Drop QC sentinel labels and configured training exclusions; return filtered AnnData and labels."""
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs")
        labels = adata.obs[label_key].astype(str)
        valid_mask = ~labels.str.strip().isin(
            ["doublet", "lowQ", "unclear", "unknown", "nan", ""]
        )
        excluded_types = self._excluded_supervised_label_strings(label_key)
        if excluded_types:
            n_before = int(valid_mask.sum())
            valid_mask = valid_mask & ~labels.str.strip().isin(excluded_types)
            n_drop = n_before - int(valid_mask.sum())
            if n_drop:
                logger.info(
                    "Excluded %s cells with %s in %r (training label filter)",
                    f"{n_drop:,}",
                    sorted(excluded_types),
                    label_key,
                )
        if copy:
            adata_filtered = adata[valid_mask].copy()
        else:
            adata_filtered = adata if bool(valid_mask.all()) else adata[valid_mask]
        labels_filtered = labels[valid_mask]
        logger.info("Filtered to %s cells with valid labels", f"{adata_filtered.n_obs:,}")
        return adata_filtered, labels_filtered

    def subset_cells_for_supervised_training(
        self,
        adata: anndata.AnnData,
        label_key: str,
        *,
        copy: bool = True,
    ) -> anndata.AnnData:
        """
        Keep cells usable for supervised training: non-missing label, QC sentinels removed,
        and ``training.exclude_hgca_celltype_v1_labels`` applied when ``label_key`` is
        ``hgca_celltype_v1``. Matches filtering used in :meth:`split_for_benchmark`.
        """
        adata_filtered, _ = self._filter_valid_label_cells(adata, label_key, copy)
        return adata_filtered

    def split_for_training(self, adata: anndata.AnnData, 
                          label_key: str,
                          test_size: float = 0.2,
                          random_state: int = 42,
                          copy: Optional[bool] = None) -> Tuple[anndata.AnnData, anndata.AnnData]:
        """
        Split data into train/test for benchmarking.
        
        NOTE: This is ONLY for training/evaluation on reference data.
        DO NOT call this for disease projection!
        
        Uses a **random stratified** split (same study can appear in both train and test).
        For leave-one-dataset-out, use ``split_for_benchmark`` with
        ``config['benchmark']['split_mode']: leave_one_dataset_out``.
        
        Args:
            adata: Preprocessed AnnData
            label_key: Column name for cell type labels
            test_size: Fraction for test set (default 0.2 = 80/20 split)
            random_state: Random seed for reproducibility
            copy: If True, materialize ``.copy()`` for filtered + train + test (three matrix copies
                peak). If False, use row **views** so train/test reference the same ``adata.X`` as
                ``adata_filtered`` (set ``data.split_copy_adata: false`` in YAML). If None, use
                ``data.split_copy_adata`` from config (default True).
            
        Returns:
            Tuple of (adata_train, adata_test)
        """
        if copy is None:
            copy = self._split_copy_adata

        logger.info(
            "Splitting data for training (test_size=%s, copy=%s)",
            test_size,
            copy,
        )
        adata_filtered, labels_filtered = self._filter_valid_label_cells(adata, label_key, copy)

        indices = np.arange(adata_filtered.n_obs)
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels_filtered,
            random_state=random_state
        )
        
        if copy:
            adata_train = adata_filtered[idx_train].copy()
            adata_test = adata_filtered[idx_test].copy()
        else:
            adata_train = adata_filtered[idx_train]
            adata_test = adata_filtered[idx_test]
            logger.info(
                "Train/test are AnnData views (no extra X copy). "
                "Avoid mutating them or the parent object; delete unneeded references to free RAM."
            )
        
        logger.info(f"✅ Split complete:")
        logger.info(f"   Train: {adata_train.n_obs:,} cells")
        logger.info(f"   Test:  {adata_test.n_obs:,} cells")
        logger.info(f"   Labels: {labels_filtered.nunique()} unique cell types")
        
        return adata_train, adata_test

    def _resolve_benchmark_dataset_column(
        self,
        bm: Dict[str, Any],
        mode: str,
        adata_filtered: anndata.AnnData,
    ) -> str:
        """
        Column used for LODO grouping and dataset-id tracking in ``split_meta``.

        If ``benchmark.dataset_id_column`` is set, use it. Otherwise, if ``obs`` has
        ``dataset_id``, use it for study-level LODO (do **not** fall back to lineage
        ``batch_key`` alone, which is often ``sample_id`` per-cell — that would be
        leave-one-sample-out, not leave-one-dataset-out).
        """
        explicit = bm.get("dataset_id_column")
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip()
        if "dataset_id" in adata_filtered.obs.columns:
            if mode == "leave_one_dataset_out":
                logger.info(
                    "benchmark.dataset_id_column unset — using 'dataset_id' for LODO (study-level)."
                )
            return "dataset_id"
        bk = str(self.config.get("batch_key") or "dataset_id")
        logger.warning(
            "obs has no 'dataset_id' column — using lineage batch_key=%r for LODO/split metadata. "
            "For study-level leave-one-dataset-out, ensure your h5ad contains dataset_id or set "
            "benchmark.dataset_id_column explicitly.",
            bk,
        )
        return bk
    
    def split_for_benchmark(
        self,
        adata: anndata.AnnData,
        label_key: str,
        copy: Optional[bool] = None,
    ) -> Tuple[anndata.AnnData, anndata.AnnData, Dict[str, Any]]:
        """
        Train/test split driven by ``config['benchmark']`` (see ``configs/scimilarity_finetune.yaml``).

        - ``split_mode: random`` — stratified random split (optimistic for unseen-dataset transfer).
        - ``split_mode: leave_one_dataset_out`` — hold out all cells with one ``dataset_id`` value
          (single LODO fold; see ``lodo_fold_index`` / ``holdout_dataset_id``).

        For **leave-one-dataset-out cross-validation** over every dataset in the grouping column,
        use :meth:`iter_split_for_benchmark_lodo` instead (one train/test pair per holdout).

        Returns ``(adata_train, adata_test, split_meta)`` where ``split_meta`` includes ``run_tag``
        for isolating outputs under ``results/.../runs/<run_tag>/``.
        """
        if copy is None:
            copy = self._split_copy_adata

        bm = dict(self.config.get("benchmark") or {})
        mode = str(bm.get("split_mode", "leave_one_dataset_out")).strip().lower()
        if mode in ("lodo", "leave_one_out"):
            mode = "leave_one_dataset_out"
        if mode not in ("random", "leave_one_dataset_out"):
            raise ValueError(f"Unknown benchmark.split_mode: {mode!r}")

        test_size = float(bm.get("test_size", 0.2))
        random_state = int(bm.get("random_state", 42))

        adata_filtered, labels_filtered = self._filter_valid_label_cells(adata, label_key, copy)

        dataset_col = self._resolve_benchmark_dataset_column(bm, mode, adata_filtered)

        if mode == "random":
            logger.info(
                "Benchmark split: random stratified (test_size=%s, random_state=%s, copy=%s)",
                test_size,
                random_state,
                copy,
            )
            indices = np.arange(adata_filtered.n_obs)
            idx_train, idx_test = train_test_split(
                indices,
                test_size=test_size,
                stratify=labels_filtered,
                random_state=random_state,
            )
            if copy:
                adata_train = adata_filtered[idx_train].copy()
                adata_test = adata_filtered[idx_test].copy()
            else:
                adata_train = adata_filtered[idx_train]
                adata_test = adata_filtered[idx_test]
                logger.info(
                    "Train/test are AnnData views (no extra X copy). "
                    "Avoid mutating them or the parent object; delete unneeded references to free RAM."
                )
            split_meta: Dict[str, Any] = {
                "split_mode": "random",
                "test_size": test_size,
                "random_state": random_state,
                "dataset_id_column": dataset_col,
                "holdout_dataset_id": None,
                "run_tag": "random_stratified",
            }
        else:
            if dataset_col not in adata_filtered.obs.columns:
                raise KeyError(
                    f"LODO requires obs column {dataset_col!r}. Set "
                    f"benchmark.dataset_id_column in YAML (or lineage batch_key) to a column that "
                    f"exists (e.g. dataset_id)."
                )
            hold_raw = bm.get("holdout_dataset_id")
            if hold_raw is None or (isinstance(hold_raw, str) and not hold_raw.strip()):
                fold_idx = int(bm.get("lodo_fold_index", 0))
                ids = np.array(
                    sorted(pd.unique(adata_filtered.obs[dataset_col].astype(str))),
                    dtype=object,
                )
                if ids.size == 0:
                    raise ValueError("No dataset ids found for leave-one-dataset-out split")
                holdout = ids[fold_idx % ids.size]
            else:
                holdout = str(hold_raw)

            logger.info(
                "Benchmark split: leave-one-dataset-out (column=%s, holdout=%s, copy=%s)",
                dataset_col,
                holdout,
                copy,
            )
            train_m, test_m = train_test_mask_leave_one_dataset(
                adata_filtered, dataset_col, holdout
            )
            if copy:
                adata_train = adata_filtered[train_m].copy()
                adata_test = adata_filtered[test_m].copy()
            else:
                adata_train = adata_filtered[train_m]
                adata_test = adata_filtered[test_m]
                logger.info(
                    "Train/test are AnnData views (no extra X copy). "
                    "Avoid mutating them or the parent object; delete unneeded references to free RAM."
                )
            safe = "".join(
                c if (c.isalnum() or c in "-_.") else "_" for c in str(holdout)
            )[:120]
            split_meta = {
                "split_mode": "leave_one_dataset_out",
                "dataset_id_column": dataset_col,
                "holdout_dataset_id": str(holdout),
                "lodo_fold_index": int(bm.get("lodo_fold_index", 0)),
                "run_tag": f"lodo_{safe}",
            }

        logger.info("✅ Split complete:")
        logger.info("   Train: %s cells", f"{adata_train.n_obs:,}")
        logger.info("   Test:  %s cells", f"{adata_test.n_obs:,}")
        logger.info("   Labels: %s unique cell types", labels_filtered.nunique())

        self._attach_split_dataset_tracking(split_meta, adata_train, adata_test, dataset_col)

        return adata_train, adata_test, split_meta

    def iter_split_for_benchmark_lodo(
        self,
        adata: anndata.AnnData,
        label_key: str,
        copy: Optional[bool] = None,
    ) -> Iterator[Tuple[anndata.AnnData, anndata.AnnData, Dict[str, Any]]]:
        """
        Leave-one-dataset-out **cross-validation**: yield one ``(train, test, split_meta)``
        per unique value in the benchmark dataset column (default ``dataset_id``).

        Uses the same filtering and column resolution as :meth:`split_for_benchmark`
        in ``leave_one_dataset_out`` mode. Respects ``benchmark.max_lodo_folds`` and
        ``benchmark.random_state`` (for subsampling folds when ``max_lodo_folds`` is set).

        Typical use: run the full benchmark once per fold and aggregate metrics
        (mean ± std over held-out datasets).
        """
        if copy is None:
            copy = self._split_copy_adata

        bm = dict(self.config.get("benchmark") or {})
        adata_filtered, _labels_filtered = self._filter_valid_label_cells(adata, label_key, copy)
        dataset_col = self._resolve_benchmark_dataset_column(
            bm, "leave_one_dataset_out", adata_filtered
        )
        if dataset_col not in adata_filtered.obs.columns:
            raise KeyError(
                f"LODO CV requires obs column {dataset_col!r}. Set benchmark.dataset_id_column "
                f"or add dataset_id to obs."
            )

        max_folds = bm.get("max_lodo_folds")
        max_folds = int(max_folds) if max_folds is not None else None
        rng = np.random.default_rng(int(bm.get("random_state", 42)))

        fold_iter = iter_leave_one_dataset_folds(
            adata_filtered,
            dataset_col,
            max_folds=max_folds,
            rng=rng,
        )

        for holdout_id, train_m, test_m in fold_iter:
            if copy:
                adata_train = adata_filtered[train_m].copy()
                adata_test = adata_filtered[test_m].copy()
            else:
                adata_train = adata_filtered[train_m]
                adata_test = adata_filtered[test_m]

            safe = "".join(
                c if (c.isalnum() or c in "-_.") else "_" for c in str(holdout_id)
            )[:120]
            split_meta: Dict[str, Any] = {
                "split_mode": "leave_one_dataset_out_cv",
                "dataset_id_column": dataset_col,
                "holdout_dataset_id": str(holdout_id),
                "run_tag": f"lodo_{safe}",
            }
            self._attach_split_dataset_tracking(
                split_meta, adata_train, adata_test, dataset_col
            )
            yield adata_train, adata_test, split_meta

    def _attach_split_dataset_tracking(
        self,
        split_meta: Dict[str, Any],
        adata_train: anndata.AnnData,
        adata_test: anndata.AnnData,
        dataset_col: str,
    ) -> None:
        """Record which dataset_ids appear in train/test (same column used for LODO when applicable)."""
        if dataset_col not in adata_train.obs.columns or dataset_col not in adata_test.obs.columns:
            split_meta["dataset_id_tracking"] = "column_missing"
            split_meta["dataset_id_column_requested"] = dataset_col
            logger.warning(
                "Split metadata: column %r not in adata.obs — cannot track dataset_ids for benchmarks",
                dataset_col,
            )
            return
        tr = adata_train.obs[dataset_col]
        te = adata_test.obs[dataset_col]
        split_meta["dataset_id_tracking"] = "ok"
        split_meta["dataset_id_column"] = dataset_col
        split_meta["n_datasets_in_train"] = int(tr.nunique())
        split_meta["n_datasets_in_test"] = int(te.nunique())
        split_meta["train_dataset_ids"] = sorted(tr.astype(str).unique().tolist())
        split_meta["test_dataset_ids"] = sorted(te.astype(str).unique().tolist())
        split_meta["test_dataset_id_counts"] = {
            str(k): int(v) for k, v in te.astype(str).value_counts().items()
        }
        split_meta["train_dataset_id_counts"] = {
            str(k): int(v) for k, v in tr.astype(str).value_counts().items()
        }
    
    def create_full_pipeline(self, label_keys: list) -> Dict:
        """
        Run full data preparation pipeline for benchmarking.
        
        This is a convenience method that combines loading, preprocessing,
        and splitting for multiple label types.
        
        Args:
            label_keys: List of label columns to benchmark
            
        Returns:
            Dictionary with prepared datasets for each label type
        """
        logger.info(f"Running full pipeline for {self.lineage}")
        
        # Load data
        adata_raw = self.load_lineage_data()
        
        # Prepare datasets for each label type
        results = {}
        
        bm = dict(self.config.get("benchmark") or {})
        mode = str(bm.get("split_mode", "leave_one_dataset_out")).strip().lower()
        if mode in ("lodo", "leave_one_out"):
            mode = "leave_one_dataset_out"
        use_lodo = mode == "leave_one_dataset_out"

        for label_key in label_keys:
            logger.info(f"\n{'='*60}")
            logger.info(f"Preparing for label type: {label_key}")
            logger.info(f"{'='*60}")
            
            # Preprocess for SCANVI
            adata_scanvi = self.preprocess(adata_raw, for_method="scanvi")
            adata_scim = self.preprocess(
                adata_raw.copy(), for_method="scimilarity", copy_adata=False
            )

            ds_col = None
            if use_lodo:
                try:
                    ds_col = self._resolve_benchmark_dataset_column(
                        bm, "leave_one_dataset_out", adata_scanvi
                    )
                except Exception:
                    ds_col = None

            if use_lodo and ds_col and ds_col in adata_scanvi.obs.columns:
                adata_train_scanvi, adata_test_scanvi, _ = self.split_for_benchmark(
                    adata_scanvi, label_key, copy=True
                )
                adata_train_scim, adata_test_scim, _ = self.split_for_benchmark(
                    adata_scim, label_key, copy=self._split_copy_adata
                )
            else:
                if use_lodo:
                    logger.warning(
                        "benchmark.split_mode is leave_one_dataset_out but dataset column %r is not "
                        "available — falling back to stratified split_for_training. "
                        "Add dataset_id to obs or set benchmark.dataset_id_column.",
                        ds_col,
                    )
                adata_train_scanvi, adata_test_scanvi = self.split_for_training(
                    adata_scanvi, label_key, copy=True
                )
                adata_train_scim, adata_test_scim = self.split_for_training(
                    adata_scim, label_key
                )
            
            results[label_key] = {
                'scanvi': {
                    'train': adata_train_scanvi,
                    'test': adata_test_scanvi
                },
                'scimilarity': {
                    'train': adata_train_scim,
                    'test': adata_test_scim
                }
            }
            
            logger.info(f"✅ Prepared datasets for {label_key}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Full pipeline complete for {self.lineage}")
        logger.info(f"{'='*60}\n")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # This would be used in a notebook like:
    # from src.utils.config import load_config
    # from src.data.preparation import DataPreparation
    #
    # config = load_config("myeloid")
    # data_prep = DataPreparation(config)
    # datasets = data_prep.create_full_pipeline(config['labels'])
    
    print("DataPreparation module loaded successfully")
    print("\nKey methods:")
    print("  - load_lineage_data(): Load 10X format data")
    print("  - preprocess(): Preprocess for specific method")
    print("  - split_for_training(): Create train/test split")
    print("  - create_full_pipeline(): Full pipeline for benchmarking")
    print("\nDesign for disease projection:")
    print("  1. Use preprocess() on disease data (no splitting!)")
    print("  2. Apply trained model.predict()")
    print("  3. Analyze predictions")


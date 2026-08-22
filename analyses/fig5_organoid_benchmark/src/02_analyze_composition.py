#!/usr/bin/env python3
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C


def clr(counts: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    logged = np.log(counts.to_numpy(float) + pseudocount)
    transformed = logged - logged.mean(axis=1, keepdims=True)
    return pd.DataFrame(transformed, index=counts.index, columns=counts.columns)


def distance_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    values = squareform(pdist(matrix.to_numpy(float), metric="euclidean"))
    return pd.DataFrame(values, index=matrix.index, columns=matrix.index)


def pcoa(distance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = distance.to_numpy(float)
    n = len(d)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-10
    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    names = [f"PCoA{i + 1}" for i in range(coordinates.shape[1])]
    coords = pd.DataFrame(coordinates, index=distance.index, columns=names)
    eig = pd.DataFrame(
        {
            "axis": [f"PCoA{i + 1}" for i in range(len(eigenvalues))],
            "eigenvalue": eigenvalues,
            "positive": eigenvalues > 0,
        }
    )
    positive_sum = eigenvalues[eigenvalues > 0].sum()
    eig["positive_variance_fraction"] = np.where(
        eig["eigenvalue"] > 0, eig["eigenvalue"] / positive_sum, 0
    )
    return coords, eig


def clr_pca(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = PCA()
    scores = model.fit_transform(matrix.to_numpy(float))
    components = model.components_.copy()
    for component in range(components.shape[0]):
        anchor = np.argmax(np.abs(components[component]))
        if components[component, anchor] < 0:
            components[component] *= -1
            scores[:, component] *= -1
    names = [f"PC{index + 1}" for index in range(scores.shape[1])]
    coordinates = pd.DataFrame(scores, index=matrix.index, columns=names)
    loadings = pd.DataFrame(
        components.T, index=matrix.columns, columns=names
    )
    loadings.index.name = "hgca_celltype_v1"
    variance = pd.DataFrame(
        {
            "axis": names,
            "explained_variance": model.explained_variance_,
            "explained_variance_fraction": model.explained_variance_ratio_,
        }
    )
    return coordinates, loadings, variance


def rarefy(counts: pd.DataFrame, depth: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sample, row in counts.iterrows():
        values = row.to_numpy(dtype=int)
        if values.sum() < depth:
            continue
        draw = rng.multivariate_hypergeometric(values, depth)
        rows.append(pd.Series(draw, index=counts.columns, name=sample))
    return pd.DataFrame(rows, columns=counts.columns).astype(int)


def lower_triangle_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 4:
        return np.nan
    av = a.loc[common, common].to_numpy()
    bv = b.loc[common, common].to_numpy()
    idx = np.tril_indices(len(common), k=-1)
    return float(spearmanr(av[idx], bv[idx]).statistic)


def one_hot(values: pd.Series) -> np.ndarray:
    frame = pd.get_dummies(values.astype(str), drop_first=True, dtype=float)
    return np.column_stack([np.ones(len(values)), frame.to_numpy(float)])


def model_r2(response: np.ndarray, design: np.ndarray) -> tuple[float, float]:
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coef
    sse = float(np.square(residual).sum())
    centered = response - response.mean(axis=0, keepdims=True)
    sst = float(np.square(centered).sum())
    return (1 - sse / sst if sst > 0 else np.nan), sse


def combined_design(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    pieces = [np.ones((len(frame), 1))]
    for field in fields:
        dummies = pd.get_dummies(
            frame[field].astype(str), prefix=field, drop_first=True, dtype=float
        )
        pieces.append(dummies.to_numpy(float))
    return np.column_stack(pieces)


def variance_partition(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variables = [
        ("publication_display", "Publication"),
        ("source_standardized", "Organoid source"),
        ("region_broad", "Broad region"),
        ("time_class", "Maturation/time"),
        ("gel", "Matrix"),
        ("molecular", "Molecular condition"),
        ("protocol", "Transplant protocol"),
        ("tech", "Sequencing protocol"),
    ]
    rows = []
    for field, display in variables:
        available = metadata[field].notna()
        index = response.index.intersection(metadata.index[available])
        frame = metadata.loc[index]
        y = response.loc[index].to_numpy(float)
        if len(index) < 8 or frame[field].nunique() < 2:
            continue
        total_r2, _ = model_r2(y, one_hot(frame[field]))
        pub_r2, pub_sse = model_r2(y, one_hot(frame["publication_display"]))
        if field == "publication_display":
            partial_r2 = total_r2
            incremental_r2 = total_r2
        else:
            _, combined_sse = model_r2(
                y, combined_design(frame, ["publication_display", field])
            )
            partial_r2 = (
                (pub_sse - combined_sse) / pub_sse if pub_sse > 0 else np.nan
            )
            intercept_sse = float(
                np.square(y - y.mean(axis=0, keepdims=True)).sum()
            )
            incremental_r2 = (
                (pub_sse - combined_sse) / intercept_sse
                if intercept_sse > 0
                else np.nan
            )

        null = []
        for _ in range(repeats):
            permuted = frame[field].copy()
            if field == "publication_display":
                permuted = pd.Series(
                    rng.permutation(permuted.to_numpy()), index=permuted.index
                )
            else:
                for _, positions in frame.groupby(
                    "publication_display", observed=True
                ).groups.items():
                    positions = list(positions)
                    permuted.loc[positions] = rng.permutation(
                        permuted.loc[positions].to_numpy()
                    )
            null_r2, _ = model_r2(y, one_hot(permuted))
            null.append(null_r2)
        null = np.asarray(null)
        p_value = (1 + np.sum(null >= total_r2 - 1e-12)) / (len(null) + 1)
        informative_publications = int(
            (
                frame.groupby("publication_display", observed=True)[field].nunique()
                > 1
            ).sum()
        )
        rows.append(
            {
                "field": field,
                "display": display,
                "n_samples": len(index),
                "n_publications": frame["publication_display"].nunique(),
                "n_levels": frame[field].nunique(),
                "n_publications_with_within_study_variation": informative_publications,
                "descriptive_r2": total_r2,
                "incremental_r2_after_publication": incremental_r2,
                "partial_r2_after_publication": partial_r2,
                "publication_r2_on_same_subset": pub_r2,
                "within_publication_permutation_p": p_value,
                "null_r2_median": float(np.median(null)),
                "null_r2_q025": float(np.quantile(null, 0.025)),
                "null_r2_q975": float(np.quantile(null, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def lopo_predict(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
) -> tuple[float, float, int, int]:
    index = response.index.intersection(metadata.index[metadata[target].notna()])
    x = response.loc[index].to_numpy(float)
    y = metadata.loc[index, target].astype(str).to_numpy()
    groups = metadata.loc[index, "publication_display"].astype(str).to_numpy()
    predictions = np.empty(len(index), dtype=object)
    valid = np.zeros(len(index), dtype=bool)
    for publication in np.unique(groups):
        test = groups == publication
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs"
            ),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        valid[test] = True
    if not valid.any():
        return np.nan, np.nan, 0, len(np.unique(groups))
    return (
        float(balanced_accuracy_score(y[valid], predictions[valid])),
        float(f1_score(y[valid], predictions[valid], average="macro")),
        int(valid.sum()),
        int(len(np.unique(groups))),
    )


def prediction_with_null(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
    repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    observed = lopo_predict(response, metadata, target)
    rng = np.random.default_rng(seed)
    null_rows = []
    for repeat in range(repeats):
        permuted = metadata.copy()
        for _, positions in permuted.groupby(
            "publication_display", observed=True
        ).groups.items():
            positions = [
                position
                for position in positions
                if pd.notna(permuted.loc[position, target])
            ]
            if len(positions) < 2:
                continue
            values = permuted.loc[positions, target].to_numpy(copy=True)
            permuted.loc[positions, target] = rng.permutation(values)
        score = lopo_predict(response, permuted, target)
        null_rows.append(
            {
                "target": target,
                "permutation": repeat,
                "balanced_accuracy": score[0],
                "macro_f1": score[1],
            }
        )
    null = pd.DataFrame(null_rows)
    valid_null = null["balanced_accuracy"].dropna()
    p = (
        (1 + (valid_null >= observed[0] - 1e-12).sum()) / (1 + len(valid_null))
        if np.isfinite(observed[0])
        else np.nan
    )
    summary = {
        "target": target,
        "balanced_accuracy": observed[0],
        "macro_f1": observed[1],
        "n_samples": observed[2],
        "n_publications": observed[3],
        "n_classes": int(metadata.loc[response.index, target].dropna().nunique()),
        "n_publications_within_study_variation": int(
            metadata.loc[
                response.index.intersection(
                    metadata.index[metadata[target].notna()]
                )
            ]
            .groupby("publication_display", observed=True)[target]
            .nunique()
            .ge(2)
            .sum()
        ),
        "null_balanced_accuracy_median": float(valid_null.median()),
        "null_balanced_accuracy_q025": float(valid_null.quantile(0.025)),
        "null_balanced_accuracy_q975": float(valid_null.quantile(0.975)),
        "within_publication_permutation_p": p,
    }
    return summary, null


def predict_missing_candidates(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
    validation_summary: dict,
) -> pd.DataFrame:
    observed = response.index.intersection(
        metadata.index[metadata[target].notna()]
    )
    missing = response.index.intersection(
        metadata.index[metadata[target].isna()]
    )
    columns = [
        "sample_id",
        "held_out_publication",
        "target",
        "predicted_value",
        "maximum_probability",
        "prediction_entropy",
        "validation_supported",
        "verification_status",
        "prediction_design",
    ]
    if len(observed) < 8 or len(missing) == 0:
        return pd.DataFrame(columns=columns)
    y = metadata.loc[observed, target].astype(str)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return pd.DataFrame(columns=columns)
    supported = bool(
        np.isfinite(validation_summary["balanced_accuracy"])
        and validation_summary["balanced_accuracy"]
        > validation_summary["null_balanced_accuracy_q975"]
        and validation_summary["within_publication_permutation_p"] < 0.05
        and validation_summary["n_publications_within_study_variation"] >= 3
    )
    rows = []
    for publication in metadata.loc[missing, "publication_display"].unique():
        test = missing[
            metadata.loc[missing, "publication_display"].eq(publication)
        ]
        train = observed[
            metadata.loc[observed, "publication_display"].ne(publication)
        ]
        train_y = metadata.loc[train, target].astype(str)
        if train_y.nunique() < 2 or len(train) < 8:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs"
            ),
        )
        model.fit(response.loc[train].to_numpy(float), train_y.to_numpy())
        probabilities = model.predict_proba(response.loc[test].to_numpy(float))
        classes = model[-1].classes_
        for sample_id, probability in zip(test, probabilities):
            rows.append(
                {
                    "sample_id": sample_id,
                    "held_out_publication": publication,
                    "target": target,
                    "predicted_value": classes[np.argmax(probability)],
                    "maximum_probability": float(probability.max()),
                    "prediction_entropy": float(
                        -np.sum(
                            probability
                            * np.log(np.clip(probability, 1e-12, 1))
                        )
                    ),
                    "validation_supported": supported,
                    "verification_status": (
                        "candidate_only_not_verified"
                        if supported
                        else "not_reportable_model_failed_validation"
                    ),
                    "prediction_design": "leave_publication_out",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def held_publication_predictions(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    observed = response.index.intersection(
        metadata.index[metadata[target].notna()]
    )
    rows = []
    for publication in metadata.loc[observed, "publication_display"].unique():
        test = observed[
            metadata.loc[observed, "publication_display"].eq(publication)
        ]
        train = observed[
            metadata.loc[observed, "publication_display"].ne(publication)
        ]
        train_y = metadata.loc[train, target].astype(str)
        if train_y.nunique() < 2 or len(train) < 8:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs"
            ),
        )
        model.fit(response.loc[train].to_numpy(float), train_y.to_numpy())
        probabilities = model.predict_proba(response.loc[test].to_numpy(float))
        classes = model[-1].classes_
        for sample_id, probability in zip(test, probabilities):
            truth = str(metadata.loc[sample_id, target])
            true_index = np.flatnonzero(classes == truth)
            rows.append(
                {
                    "sample_id": sample_id,
                    "held_out_publication": publication,
                    "target": target,
                    "true_value": truth,
                    "predicted_value": classes[np.argmax(probability)],
                    "correct": classes[np.argmax(probability)] == truth,
                    "maximum_probability": float(probability.max()),
                    "true_class_probability": (
                        float(probability[true_index[0]])
                        if len(true_index)
                        else np.nan
                    ),
                    "prediction_design": "leave_publication_out",
                }
            )
    return pd.DataFrame(rows)


def publication_centered_enrichment(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    fields: list[str],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for field in fields:
        index = response.index.intersection(metadata.index[metadata[field].notna()])
        if len(index) < 6:
            continue
        values = metadata.loc[index, field].astype(str)
        publications = metadata.loc[index, "publication_display"].astype(str)
        centered = response.loc[index] - response.loc[index].groupby(
            publications, observed=True
        ).transform("mean")
        scale = centered.std(axis=0, ddof=1).replace(0, np.nan)
        publication_variation = (
            values.groupby(publications, observed=True).nunique().ge(2)
        )
        variable_publications = publication_variation.index[
            publication_variation
        ]
        eligible = publications.isin(variable_publications)
        for category in sorted(values.unique()):
            category_mask = values.eq(category)
            analysis_mask = eligible & (
                publications.isin(
                    publications[eligible & category_mask].unique()
                )
            )
            group_mask = analysis_mask & category_mask
            comparison_mask = analysis_mask & ~category_mask
            if group_mask.sum() == 0 or comparison_mask.sum() == 0:
                effect = pd.Series(np.nan, index=response.columns)
                p_values = pd.Series(np.nan, index=response.columns)
            else:
                effect = (
                    centered.loc[group_mask].mean()
                    - centered.loc[comparison_mask].mean()
                ) / scale
                null = []
                analysis_index = index[analysis_mask]
                analysis_publications = publications.loc[analysis_index]
                analysis_values = values.loc[analysis_index]
                for _ in range(permutations):
                    permuted = analysis_values.copy()
                    for publication, positions in analysis_publications.groupby(
                        analysis_publications, observed=True
                    ).groups.items():
                        permuted.loc[positions] = rng.permutation(
                            permuted.loc[positions].to_numpy()
                        )
                    permuted_group = permuted.eq(category)
                    if permuted_group.all() or (~permuted_group).all():
                        continue
                    null.append(
                        (
                            centered.loc[permuted.index[permuted_group]].mean()
                            - centered.loc[permuted.index[~permuted_group]].mean()
                        ).div(scale)
                        .to_numpy(float)
                    )
                null_array = np.asarray(null)
                p_values = pd.Series(
                    (
                        1
                        + (np.abs(null_array) >= np.abs(effect.to_numpy())).sum(
                            axis=0
                        )
                    )
                    / (len(null_array) + 1),
                    index=response.columns,
                )
            descriptive = (
                response.loc[index[category_mask]].mean()
                - response.loc[index[~category_mask]].mean()
            ) / response.loc[index].std(axis=0, ddof=1).replace(0, np.nan)
            n_variable = int(
                publications.loc[group_mask].nunique()
                if group_mask.any()
                else 0
            )
            for subtype in response.columns:
                rows.append(
                    {
                        "field": field,
                        "category": category,
                        "hgca_celltype_v1": subtype,
                        "n_category_samples": int(category_mask.sum()),
                        "n_category_publications": int(
                            publications.loc[category_mask].nunique()
                        ),
                        "n_variable_publications": n_variable,
                        "descriptive_standardized_enrichment": descriptive[
                            subtype
                        ],
                        "publication_centered_standardized_enrichment": effect[
                            subtype
                        ],
                        "within_publication_permutation_p": p_values[subtype],
                        "estimable_across_publications": n_variable >= 2,
                    }
                )
    result = pd.DataFrame(rows)
    result["within_publication_permutation_q"] = np.nan
    for field, positions in result.groupby("field", observed=True).groups.items():
        p = result.loc[positions, "within_publication_permutation_p"]
        valid = p.notna()
        if not valid.any():
            continue
        order = p.loc[valid].sort_values().index
        ranks = np.arange(1, len(order) + 1)
        adjusted = np.minimum.accumulate(
            (p.loc[order].to_numpy() * len(order) / ranks)[::-1]
        )[::-1]
        result.loc[order, "within_publication_permutation_q"] = np.minimum(
            adjusted, 1
        )
    return result


def adjusted_covariate_enrichment(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = {
        "source_standardized": ["region_broad", "time_class"],
        "region_broad": ["source_standardized", "time_class"],
        "time_class": ["source_standardized", "region_broad"],
        "gel": ["source_standardized", "region_broad", "time_class"],
        "molecular": [
            "source_standardized",
            "region_broad",
            "time_class",
            "gel",
        ],
    }
    summaries = []
    residual_frames = []
    for target, adjustments in specifications.items():
        required = [target, *adjustments]
        available = metadata[required].notna().all(axis=1)
        index = response.index.intersection(metadata.index[available])
        frame = metadata.loc[index, required]
        if len(index) < 8 or frame[target].nunique() < 2:
            continue
        design = combined_design(frame, adjustments)
        y = response.loc[index].to_numpy(float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        scale = residual.std(axis=0, ddof=1)
        scale[scale == 0] = np.nan
        standardized = residual / scale
        residual_frame = pd.DataFrame(
            standardized, index=index, columns=response.columns
        )
        residual_frame.insert(0, "target_value", frame[target].astype(str))
        residual_frame.insert(0, "target", target)
        residual_frame.index.name = "sample_id"
        residual_frames.append(residual_frame.reset_index())
        for category in sorted(frame[target].astype(str).unique()):
            category_mask = frame[target].astype(str).eq(category).to_numpy()
            comparison_mask = ~category_mask
            if category_mask.sum() < 2 or comparison_mask.sum() < 2:
                continue
            effect = (
                np.nanmean(standardized[category_mask], axis=0)
                - np.nanmean(standardized[comparison_mask], axis=0)
            )
            for subtype_index, subtype in enumerate(response.columns):
                summaries.append(
                    {
                        "field": target,
                        "category": category,
                        "hgca_celltype_v1": subtype,
                        "adjusted_for": ";".join(adjustments),
                        "n_complete_samples": len(index),
                        "n_category_samples": int(category_mask.sum()),
                        "n_publications": metadata.loc[
                            index, "publication_display"
                        ].nunique(),
                        "adjusted_standardized_enrichment": effect[
                            subtype_index
                        ],
                    }
                )
    return pd.DataFrame(summaries), pd.concat(
        residual_frames, ignore_index=True
    )


def pca_partial_r2(
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    coordinate_variance = coordinates.var(axis=0, ddof=1)
    nonzero_pcs = coordinate_variance[
        coordinate_variance
        > coordinate_variance.max() * 1e-12
    ].index.tolist()
    if not {"PC1", "PC2"}.issubset(nonzero_pcs):
        raise RuntimeError("CLR-PCA coordinates do not contain non-zero PC1 and PC2")
    pc1_pc2_variance_fraction = float(
        coordinate_variance.loc[["PC1", "PC2"]].sum()
        / coordinate_variance.loc[nonzero_pcs].sum()
    )
    specifications = {
        "publication_display": [
            "source_standardized",
            "region_broad",
            "time_class",
        ],
        "source_standardized": ["region_broad", "time_class"],
        "region_broad": ["source_standardized", "time_class"],
        "time_class": ["source_standardized", "region_broad"],
        "gel": ["source_standardized", "region_broad", "time_class"],
        "molecular": [
            "source_standardized",
            "region_broad",
            "time_class",
            "gel",
        ],
        "tech": ["source_standardized", "region_broad", "time_class"],
    }
    display = {
        "publication_display": "Publication",
        "source_standardized": "Organoid source",
        "region_broad": "Region",
        "time_class": "Maturation/time",
        "gel": "Matrix",
        "molecular": "Molecular condition",
        "tech": "Sequencing protocol",
    }
    rows = []
    for target, adjustments in specifications.items():
        required = [target, *adjustments]
        available = metadata[required].notna().all(axis=1)
        index = coordinates.index.intersection(metadata.index[available])
        frame = metadata.loc[index, required]
        if len(index) < 8 or frame[target].nunique() < 2:
            continue
        y_all = coordinates.loc[index, nonzero_pcs].to_numpy(float)
        y_pc1_pc2 = coordinates.loc[index, ["PC1", "PC2"]].to_numpy(float)
        base = combined_design(frame, adjustments)
        full = combined_design(frame, [*adjustments, target])
        _, base_sse_all = model_r2(y_all, base)
        _, full_sse_all = model_r2(y_all, full)
        _, base_sse_pc1_pc2 = model_r2(y_pc1_pc2, base)
        _, full_sse_pc1_pc2 = model_r2(y_pc1_pc2, full)
        rows.append(
            {
                "field": target,
                "display": display[target],
                "adjusted_for": ";".join(adjustments),
                "n_samples": len(index),
                "n_levels": frame[target].nunique(),
                "n_nonzero_pcs": len(nonzero_pcs),
                "all_pcs_variance_fraction": 1.0,
                "pc1_pc2_variance_fraction": pc1_pc2_variance_fraction,
                "base_design_rank": int(np.linalg.matrix_rank(base)),
                "full_design_rank": int(np.linalg.matrix_rank(full)),
                "partial_r2_all_nonzero_pcs": (
                    (base_sse_all - full_sse_all) / base_sse_all
                    if base_sse_all > 0
                    else np.nan
                ),
                "partial_r2_pc1_pc2": (
                    (base_sse_pc1_pc2 - full_sse_pc1_pc2)
                    / base_sse_pc1_pc2
                    if base_sse_pc1_pc2 > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_counts(
    counts: pd.DataFrame, hierarchy: pd.DataFrame, level: str
) -> pd.DataFrame:
    mapping = hierarchy[level].reindex(counts.columns)
    mapping = mapping.where(mapping.notna(), counts.columns.to_series(index=counts.columns))
    result = {}
    for group, labels in mapping.groupby(mapping).groups.items():
        result[str(group)] = counts.loc[:, list(labels)].sum(axis=1)
    return pd.DataFrame(result, index=counts.index)


def sensitivity_analysis(
    counts: pd.DataFrame,
    hierarchy: pd.DataFrame,
    config: dict,
    primary_distance: pd.DataFrame,
) -> pd.DataFrame:
    seed = int(config["project"]["seed"])
    rows = []
    depths = config["filters"]["sensitivity_min_confident_cells"]
    pseudocounts = config["filters"]["sensitivity_pseudocounts"]
    for depth, pseudocount, level, drop_rare in itertools.product(
        depths,
        pseudocounts,
        ["hgca_celltype_v1", "hgca_celltype_level2", "hgca_celltype_level3"],
        [False, True],
    ):
        current = counts
        if level != "hgca_celltype_v1":
            current = aggregate_counts(current, hierarchy, level)
        if drop_rare:
            keep = current.sum(axis=0) >= int(
                config["filters"]["rare_subtype_min_total_cells"]
            )
            current = current.loc[:, keep]
        rarefied = rarefy(current, int(depth), seed + int(depth))
        if len(rarefied) < 4 or rarefied.shape[1] < 2:
            continue
        transformed = clr(rarefied, float(pseudocount))
        distance = distance_frame(transformed)
        rows.append(
            {
                "min_confident_cells": depth,
                "pseudocount": pseudocount,
                "annotation_level": level,
                "rare_subtypes_excluded": drop_rare,
                "n_samples": len(rarefied),
                "n_features": rarefied.shape[1],
                "distance_spearman_vs_primary": lower_triangle_correlation(
                    primary_distance, distance
                ),
            }
        )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    prop_distance = distance_frame(proportions)
    rows.append(
        {
            "min_confident_cells": 0,
            "pseudocount": np.nan,
            "annotation_level": "raw_proportions",
            "rare_subtypes_excluded": False,
            "n_samples": len(proportions),
            "n_features": proportions.shape[1],
            "distance_spearman_vs_primary": lower_triangle_correlation(
                primary_distance, prop_distance
            ),
        }
    )
    return pd.DataFrame(rows)


def loadings_and_clusters(
    transformed: pd.DataFrame,
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loading_rows = []
    for axis in ["PC1", "PC2"]:
        for subtype in transformed.columns:
            result = spearmanr(
                transformed[subtype], coordinates.loc[transformed.index, axis]
            )
            loading_rows.append(
                {
                    "axis": axis,
                    "hgca_celltype_v1": subtype,
                    "spearman_r": result.statistic,
                    "p_value": result.pvalue,
                }
            )
    loadings = pd.DataFrame(loading_rows)

    values = coordinates[["PC1", "PC2"]].to_numpy(float)
    candidates = []
    for k in range(2, min(7, len(values) - 1)):
        labels = KMeans(n_clusters=k, random_state=0, n_init=50).fit_predict(values)
        candidates.append((silhouette_score(values, labels), k, labels))
    _, best_k, labels = max(candidates, key=lambda item: item[0])
    clusters = pd.DataFrame(
        {"composition_cluster": [f"C{x + 1}" for x in labels]},
        index=coordinates.index,
    ).join(metadata)
    enrichment = transformed.groupby(
        clusters["composition_cluster"], observed=True
    ).mean()
    enrichment = enrichment.subtract(transformed.mean(axis=0), axis=1)
    enrichment.index.name = "composition_cluster"
    return loadings, clusters, enrichment


def main() -> None:
    config = C.load_config()
    C.require_files(config)
    seed = int(config["project"]["seed"])
    C.set_seed(seed)
    logger = C.setup_logging("02_analyze_composition")

    counts = pd.read_csv(
        C.DATA / "sample_subtype_counts_confident.csv", index_col=0
    )
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv", index_col=0
    )
    depth = int(config["filters"]["primary_min_confident_cells"])
    rarefied = rarefy(counts, depth, seed)
    transformed = clr(rarefied, pseudocount=1.0)
    distance = distance_frame(transformed)
    coordinates, pca_loadings, pca_variance = clr_pca(transformed)

    transformed.to_csv(C.DATA / "fig5c_clr_composition.csv")
    distance.to_csv(C.DATA / "fig5c_aitchison_distance.csv")
    coordinates.to_csv(C.DATA / "fig5c_clr_pca_coordinates.csv")
    pca_loadings.to_csv(C.DATA / "fig5c_clr_pca_loadings.csv")
    pca_variance.to_csv(C.DATA / "fig5c_clr_pca_variance.csv", index=False)

    variance = variance_partition(
        transformed,
        metadata,
        int(config["statistics"]["permutation_repeats"]),
        seed,
    )
    variance.to_csv(C.DATA / "fig5c_variance_partition.csv", index=False)

    prediction_summaries = []
    prediction_nulls = []
    missing_candidates = []
    heldout_predictions = []
    target_map = {
        "derive": "source_standardized",
        "tissue": "region_detail",
        "time": "time_class",
        "matrix": "gel",
    }
    for configured in config["statistics"]["prediction_targets"]:
        target = target_map[configured]
        summary, null = prediction_with_null(
            transformed,
            metadata,
            target,
            int(config["statistics"]["permutation_repeats"]),
            seed,
        )
        prediction_summaries.append(summary)
        prediction_nulls.append(null)
        missing_candidates.append(
            predict_missing_candidates(transformed, metadata, target, summary)
        )
        heldout_predictions.append(
            held_publication_predictions(transformed, metadata, target)
        )
    prediction_summary_frame = pd.DataFrame(prediction_summaries)
    prediction_summary_frame.to_csv(
        C.DATA / "supp_metadata_prediction_summary.csv", index=False
    )
    pd.concat(prediction_nulls, ignore_index=True).to_csv(
        C.DATA / "supp_metadata_prediction_null.csv", index=False
    )
    candidate_frames = [
        frame for frame in missing_candidates if not frame.empty
    ]
    pd.concat(candidate_frames, ignore_index=True).to_csv(
        C.DATA / "supp_missing_metadata_prediction_candidates.csv", index=False
    )
    heldout_frame = pd.concat(heldout_predictions, ignore_index=True)
    heldout_frame.to_csv(
        C.DATA / "supp_metadata_prediction_heldout_samples.csv", index=False
    )
    (
        heldout_frame.groupby(
            ["target", "held_out_publication"], observed=True
        )
        .agg(
            n_test_samples=("sample_id", "size"),
            fold_accuracy=("correct", "mean"),
            median_maximum_probability=("maximum_probability", "median"),
            median_true_class_probability=("true_class_probability", "median"),
        )
        .reset_index()
        .to_csv(
            C.DATA / "supp_metadata_prediction_heldout_folds.csv",
            index=False,
        )
    )
    enrichment = publication_centered_enrichment(
        transformed,
        metadata,
        [
            "source_standardized",
            "region_broad",
            "time_class",
            "gel",
            "molecular",
            "protocol",
        ],
        int(config["statistics"]["permutation_repeats"]),
        seed,
    )
    enrichment.to_csv(
        C.DATA / "supp_covariate_subtype_enrichment.csv", index=False
    )
    adjusted_enrichment, adjusted_residuals = adjusted_covariate_enrichment(
        transformed, metadata
    )
    adjusted_enrichment.to_csv(
        C.DATA / "supp_covariate_subtype_adjusted_enrichment.csv",
        index=False,
    )
    adjusted_residuals.to_csv(
        C.DATA / "supp_covariate_subtype_adjusted_residuals.csv.gz",
        index=False,
        compression="gzip",
    )
    pca_partial_r2(coordinates, metadata).to_csv(
        C.DATA / "fig5c_pca_partial_r2.csv", index=False
    )
    eligibility = pd.DataFrame(
        [
            {
                "target": "Organoid source",
                "tested": True,
                "n_samples": int(metadata.loc[transformed.index, "source_standardized"].notna().sum()),
                "reason": "Complete; evaluated by leave-one-publication-out CV with a within-publication permutation null.",
            },
            {
                "target": "Detailed region",
                "tested": True,
                "n_samples": int(metadata.loc[transformed.index, "region_detail"].notna().sum()),
                "reason": "Three standardized regions; nonspecific intestine samples excluded.",
            },
            {
                "target": "Publication",
                "tested": False,
                "n_samples": len(transformed),
                "reason": "Publication is a nuisance/confounding variable and cannot be leave-one-publication-out predicted as its own target.",
            },
            {
                "target": "Maturation/time",
                "tested": True,
                "n_samples": int(metadata.loc[transformed.index, "time_class"].notna().sum()),
                "reason": "Exploratory LOPO test; time definitions are publication-specific and only two publications contain within-study variation after harmonization.",
            },
            {
                "target": "Transplanted versus in vitro",
                "tested": False,
                "n_samples": int(metadata.loc[transformed.index, "protocol"].notna().sum()),
                "reason": "Only transplant-positive samples are encoded; the missing values are not verified in-vitro controls.",
            },
            {
                "target": "Matrix",
                "tested": True,
                "n_samples": int(metadata.loc[transformed.index, "gel"].notna().sum()),
                "reason": "Exploratory LOPO test; incomplete and strongly study-specific.",
            },
            {
                "target": "Molecular condition",
                "tested": False,
                "n_samples": int(metadata.loc[transformed.index, "molecular"].notna().sum()),
                "reason": "Seven heterogeneous levels across six publications; labels are not commensurate across studies.",
            },
        ]
    )
    eligibility.to_csv(C.DATA / "supp_prediction_target_eligibility.csv", index=False)

    sensitivity = sensitivity_analysis(
        counts, hierarchy, config, distance
    )
    strict_path = C.DATA / "sample_subtype_counts_strict_mapping.csv"
    if strict_path.is_file():
        strict_counts = pd.read_csv(strict_path, index_col=0)
        strict_counts = strict_counts.reindex(
            columns=counts.columns, fill_value=0
        )
        strict_rarefied = rarefy(strict_counts, depth, seed)
        if len(strict_rarefied) >= 4:
            strict_distance = distance_frame(clr(strict_rarefied, 1.0))
            sensitivity = pd.concat(
                [
                    sensitivity,
                    pd.DataFrame(
                        [
                            {
                                "min_confident_cells": depth,
                                "pseudocount": 1.0,
                                "annotation_level": "strict_confidence_plus_distance",
                                "rare_subtypes_excluded": False,
                                "n_samples": len(strict_rarefied),
                                "n_features": strict_rarefied.shape[1],
                                "distance_spearman_vs_primary": lower_triangle_correlation(
                                    distance, strict_distance
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    sensitivity.to_csv(C.DATA / "supp_embedding_sensitivity.csv", index=False)

    loadings, clusters, enrichment = loadings_and_clusters(
        transformed, coordinates, metadata
    )
    loadings.to_csv(C.DATA / "fig5c_subtype_axis_associations.csv", index=False)
    clusters.to_csv(C.DATA / "fig5c_sample_clusters.csv")
    enrichment.to_csv(C.DATA / "fig5c_cluster_subtype_enrichment.csv")

    complete = metadata.loc[transformed.index].copy()
    fields = ["publication_display", "source_standardized", "region_broad"]
    complete = complete.dropna(subset=fields)
    y = transformed.loc[complete.index].to_numpy(float)
    full_r2, _ = model_r2(y, combined_design(complete, fields))
    publication_r2, _ = model_r2(
        y, combined_design(complete, ["publication_display"])
    )
    summary = pd.DataFrame(
        [
            {
                "model": "Publication",
                "n_samples": len(complete),
                "explained_fraction": publication_r2,
                "unexplained_fraction": 1 - publication_r2,
            },
            {
                "model": "Publication + source + broad region",
                "n_samples": len(complete),
                "explained_fraction": full_r2,
                "unexplained_fraction": 1 - full_r2,
            },
        ]
    )
    summary.to_csv(C.DATA / "supp_explained_unexplained_structure.csv", index=False)

    axis_covariates = []
    joined = coordinates.join(metadata)
    joined = joined.join(
        pd.read_csv(C.DATA / "sample_mapping_qc.csv", index_col=0)[
            ["n_confident_sysvi", "fraction_confident_sysvi"]
        ]
    )
    for axis in ["PC1", "PC2"]:
        for covariate in ["n_confident_sysvi", "fraction_confident_sysvi"]:
            result = spearmanr(joined[axis], joined[covariate])
            axis_covariates.append(
                {
                    "axis": axis,
                    "covariate": covariate,
                    "spearman_r": result.statistic,
                    "p_value": result.pvalue,
                    "n_samples": len(joined),
                }
            )
    pd.DataFrame(axis_covariates).to_csv(
        C.DATA / "supp_axis_depth_qc_associations.csv", index=False
    )
    logger.info(
        "Primary composition: %s samples rarefied to %s confident cells; "
        "PC1 %.1f%%, PC2 %.1f%% CLR variance",
        len(rarefied),
        depth,
        100 * pca_variance.loc[0, "explained_variance_fraction"],
        100 * pca_variance.loc[1, "explained_variance_fraction"],
    )


if __name__ == "__main__":
    main()

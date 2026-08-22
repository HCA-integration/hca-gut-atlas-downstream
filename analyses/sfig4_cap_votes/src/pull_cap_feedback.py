"""
Pull expert feedback + label metadata for the HGCA CAP project (project 901)
directly from the Cell Annotation Platform public GraphQL API
(https://celltype.info/graphql).

The API only accepts persisted (safelisted) queries, so we use the same
Automatic-Persisted-Query (APQ) protocol the official cap-sc-client uses:
we send the "DatasetDataQuery" operation by its sha256 hash. That query returns,
for each labelset -> label:
  - CAP label metadata (markerGenes, canonicalMarkerGenes, rationale, synonyms,
    ontology term)
  - every expert feedback item (score, agree/disagree/split/merge/refine/idk,
    the free-text comment)

We pull the four HGCA lineage datasets and keep the "Prelim annotation" labelset,
which is the annotation set the CAP experts reviewed.

Outputs (analyses/sfig4_cap_votes/out/ unless CAP_OUT is set):
  - cap_labels_901.csv     one row per label with CAP metadata + score summary
  - cap_feedback_901.csv   one row per feedback item (long format)
  - cap_feedback_901.json  full raw payload per dataset (for auditing)

Tracked fixtures under data/cap/ are a redacted snapshot (no user IDs
or free-text comments) plus the label summary.
"""
import csv
import json
import os
import subprocess
from pathlib import Path

API = "https://celltype.info/graphql"
DATASET_DATA_QUERY_HASH = (
    "0bf580de5e51d602a103067cf8cab04d4f7f7ece914e95e02ee17a2b5b6e403a"
)
# HGCA lineage datasets in CAP project 901
DATASETS = {
    "2166": "epithelial",
    "2167": "myeloid",
    "2168": "stromal",
    "2169": "lymphoid",
}
LABELSET = "Prelim annotation"

OUT = Path(
    os.environ.get(
        "CAP_OUT",
        str(Path(__file__).resolve().parents[1] / "out"),
    )
)
OUT.mkdir(parents=True, exist_ok=True)


def fetch_dataset(dataset_id):
    payload = json.dumps({
        "operationName": "DatasetDataQuery",
        "variables": {"datasetId": dataset_id},
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": DATASET_DATA_QUERY_HASH}},
    })
    proc = subprocess.run(
        ["curl", "-s", "-m", "180", "-X", "POST", API,
         "-H", "Content-Type: application/json", "--data", payload],
        capture_output=True, text=True,
    )
    data = json.loads(proc.stdout)
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"])[:2000])
    return data["data"]["dataset"]


def explanation_to_text(expl):
    """Return (type, comment, refine_changes) for an explanation node."""
    if not expl:
        return "", "", ""
    etype = expl.get("type") or ""
    d = expl.get("data") or {}
    comment = d.get("comment") or ""
    refine = ""
    for ch in (d.get("changes") or []):
        refine += f"[{ch.get('attribute')}] {ch.get('originalValue')} -> {ch.get('newValue')}; "
    return etype, comment, refine.strip()


def main():
    raw = {}
    label_rows = []
    fb_rows = []
    for ds_id, lineage in DATASETS.items():
        ds = fetch_dataset(ds_id)
        raw[ds_id] = ds
        ds_name = ds["name"]
        labelsets = [ls for ls in ds["labelsets"] if ls["name"] == LABELSET]
        if not labelsets:
            raise RuntimeError(f"No '{LABELSET}' labelset in dataset {ds_id}")
        for ls in labelsets:
            for lab in ls["labels"]:
                scores = lab.get("scores") or {}
                fbs = lab.get("feedbacks") or []
                label_rows.append({
                    "lineage": lineage,
                    "dataset_id": ds_id,
                    "dataset_name": ds_name,
                    "label_id": lab["id"],
                    "label_name": lab["name"],
                    "full_name": lab.get("fullName") or "",
                    "cell_count": lab.get("count") or "",
                    "cap_marker_genes": ",".join(lab.get("markerGenes") or []),
                    "cap_canonical_marker_genes": ",".join(lab.get("canonicalMarkerGenes") or []),
                    "cap_synonyms": ",".join(lab.get("synonyms") or []),
                    "cap_rationale": (lab.get("rationale") or "").replace("\n", " "),
                    "cap_rationale_dois": ",".join(lab.get("rationaleDois") or []),
                    "cap_ontology_term": lab.get("ontologyTerm") or "",
                    "cap_ontology_term_id": lab.get("ontologyTermId") or "",
                    "score_agree": scores.get("agree"),
                    "score_disagree": scores.get("disagree"),
                    "score_idk": scores.get("idk"),
                    "n_feedback": len(fbs),
                })
                for fb in fbs:
                    etype, comment, refine = explanation_to_text(fb.get("explanation"))
                    user = fb.get("user") or {}
                    fb_rows.append({
                        "lineage": lineage,
                        "dataset_id": ds_id,
                        "label_id": lab["id"],
                        "label_name": lab["name"],
                        "score": fb.get("score"),
                        "explanation_type": etype,
                        "createdAt": fb.get("createdAt") or "",
                        "isUpdated": fb.get("isUpdated"),
                        "user_uid": user.get("uid") or "",
                        "comment": comment,
                        "refine_changes": refine,
                    })

    (OUT / "cap_feedback_901.json").write_text(json.dumps(raw, indent=2))

    lp = OUT / "cap_labels_901.csv"
    with open(lp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        w.writeheader()
        w.writerows(label_rows)

    fp = OUT / "cap_feedback_901.csv"
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fb_rows[0].keys()))
        w.writeheader()
        w.writerows(fb_rows)

    print(f"labels: {len(label_rows)}  feedback items: {len(fb_rows)}")
    for ds_id, lineage in DATASETS.items():
        n_lab = sum(1 for r in label_rows if r["dataset_id"] == ds_id)
        n_fb = sum(1 for r in fb_rows if r["dataset_id"] == ds_id)
        print(f"  {lineage} (ds {ds_id}): {n_lab} labels, {n_fb} feedback")
    print("wrote:", lp)
    print("wrote:", fp)
    print("wrote:", OUT / "cap_feedback_901.json")


if __name__ == "__main__":
    main()

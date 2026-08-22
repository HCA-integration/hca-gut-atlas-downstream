# Supplementary Figure 4 — CAP votes

GraphQL pull of expert feedback on the HGCA v0 (“Prelim annotation”)
labelset in CAP project 901 (datasets 2166–2169).

Script: `src/pull_cap_feedback.py` (public `https://celltype.info/graphql`
APQ; no login).

Tracked fixtures under `data/cap/`:

| File | Contents |
|---|---|
| `cap_labels_901.csv` | Per-label agree / disagree / idk and feedback counts |
| `cap_feedback_901.csv` | One row per vote; `explanation_type` only (no user IDs or comments) |
| `taxonomy_marker_source_report.csv` | v1 ↔ CAP label bridge used by Figure 2 |

```bash
python analyses/sfig4_cap_votes/src/pull_cap_feedback.py
```

Figure 2 reads the fixtures by default (`HGCA_CAP_LABELS`,
`HGCA_CAP_FEEDBACK`, `HGCA_CAP_BRIDGE` override). `--demo` still skips
CAP. A live re-pull writes the full payload, including comments, under
`out/`.

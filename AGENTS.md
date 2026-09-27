# Workspace Guidelines & Instructions

## Submission Artifacts Rules
- **Do NOT generate `candidate_pairs` (e.g. `candidate_pairs_V2.tsv` or `candidate_pairs.tsv`) by default** during pipeline runs, training, or evaluation.
- `candidate_pairs` is only needed when preparing the final submission package.
- Only generate `matching_results_V2.tsv` by default.
- Only generate `candidate_pairs` when the user explicitly requests it or passes `--save-candidates` / `--generate-candidates`.
- Always store submission results in `output/submmision/`.

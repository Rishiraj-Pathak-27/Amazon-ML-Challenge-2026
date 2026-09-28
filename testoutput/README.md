# Unstop Portal Simulation & Benchmark Suite

This directory contains the automated evaluation and simulation suite for the **Amazon ML Challenge 2026** entity resolution challenge.

## Contents
- [`simulate_unstop_portal.py`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/testouput/simulate_unstop_portal.py): Core test script simulating the Unstop Portal validation checks, F0.5 scoring engine, Public/Private leaderboard splits, and full output file benchmark.
- [`test_all_outputs.py`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/testouput/test_all_outputs.py): Symlink entrypoint to `simulate_unstop_portal.py`.
- [`hidden_testcases.tsv`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/testouput/hidden_testcases.tsv): 30,000 real ground-truth evaluation entities extracted from training ground truth for offline validation.
- [`leaderboard_summary.json`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/testouput/leaderboard_summary.json): Detailed machine-readable JSON leaderboard ranking.

---

## Benchmark Results Summary

### Official Hidden Testcase Leaderboard (5,000 Common Benchmark Cases)

| Rank | Output File | Public (50%) | Private (50%) | Overall F0.5 | Precision (2x) | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | [`output/submmision/matching_results_V16.tsv`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/output/submmision/matching_results_V16.tsv) | **0.76804** | **0.77650** | **0.77227** | **0.8901** | 0.5840 |
| 🥈 **2** | [`output/submmision/matching_results_V17.tsv`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/output/submmision/matching_results_V17.tsv) | 0.76437 | 0.77047 | **0.76742** | 0.8861 | 0.5780 |
| 🥉 **3** | [`output/submmision/matching_results_V14.tsv`](file:///Users/utkarshpatrikar/Code%20Files/business_entity_resolution/output/submmision/matching_results_V14.tsv) | 0.74786 | 0.75718 | **0.75252** | 0.8657 | 0.5764 |
| 4 | `output/submmision/matching_results_V11.tsv` | 0.73939 | 0.75125 | **0.74532** | 0.8217 | 0.6182 |
| 5 | `output/submmision/matching_results_V12.tsv` | 0.73875 | 0.75166 | **0.74520** | 0.8218 | 0.6176 |
| 6 | `output/submmision/matching_results_V15.tsv` | 0.73080 | 0.74123 | **0.73602** | 0.8363 | 0.5801 |
| 7 | `output/submmision/matching_results_V10.tsv` | 0.71907 | 0.72171 | **0.72039** | 0.8081 | 0.5749 |
| 8 | `output/submmision/matching_results_V13.tsv` | 0.69956 | 0.71437 | **0.70697** | 0.7972 | 0.5816 |
| 9 | `output/submmision/matching_results_V8.tsv` | 0.67138 | 0.67176 | **0.67157** | 0.7157 | 0.6111 |
| 10 | `output/submmision/matching_results_V9.tsv` | 0.67045 | 0.67031 | **0.67038** | 0.7135 | 0.6125 |
| 11 | `outputs/matching_results_zeba_V1.tsv` | 0.06035 | 0.05485 | **0.05760** | 0.0588 | 0.0563 |

---

## How to Run

### Test All Output Files
```bash
python3 testouput/simulate_unstop_portal.py
# or
python3 testouput/test_all_outputs.py
```

### Test a Specific Output File
```bash
python3 testouput/simulate_unstop_portal.py --file output/submmision/matching_results_V16.tsv
```

### Evaluate on a Custom Benchmark Size (e.g., 10,000 cases)
```bash
python3 testouput/simulate_unstop_portal.py --benchmark-size 10000
```

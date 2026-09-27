#!/usr/bin/env python3
"""
================================================================================
UNSTOP SUBMISSION EVALUATION PORTAL — AMAZON ML CHALLENGE 2026
Automated Submission Validation & Hidden Testcase Scoring Engine

Simulates the exact Unstop leaderboard evaluation portal:
  1. Validates all submission constraints & formatting rules (TSV, headers, prefixes,
     self-matches, duplicates, 512 MB file limit).
  2. Evaluates against hidden ground-truth testcases using official Macro F0.5 metric.
  3. Splits evaluation into Public Leaderboard (50%) & Private Leaderboard (50%).
  4. Automatically scans all output files across output directories.
  5. Ranks every candidate and identifies the BEST SCORING FILE.
================================================================================
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Set, List, Tuple, Optional

# ANSI Color formatting
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"

DELIM = "\t"
EXPECTED_HEADER = ["source1_entity_id", "matched_entity_ids"]
MAX_PORTAL_SIZE_MB = 512.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_HIDDEN_PATH = os.path.join(SCRIPT_DIR, "hidden_testcases.tsv")


def print_banner():
    banner = f"""
{CYAN}╔════════════════════════════════════════════════════════════════════════════════════════╗
║             {YELLOW}UNSTOP EVALUATION PORTAL SIMULATOR — AMAZON ML CHALLENGE 2026{CYAN}              ║
║                 Automated Verification & Hidden Testcase Benchmarking                  ║
╚════════════════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def compute_entity_f_beta(pred_ids: Set[str], true_ids: Set[str], beta: float = 0.5) -> Tuple[float, float, float]:
    """
    Score a single Source-1 entity prediction against ground truth.
    Returns: (f_beta, precision, recall)
    """
    if not true_ids and not pred_ids:
        return 1.0, 1.0, 1.0  # Correctly predicted singleton
    if not pred_ids:
        return 0.0, 0.0, 0.0  # False negative (missed match)
    if not true_ids:
        return 0.0, 0.0, 0.0  # False positive (false merge on singleton)

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        f_beta = 0.0
    else:
        f_beta = (1.0 + beta2) * precision * recall / denom

    return f_beta, precision, recall


def load_hidden_ground_truth(path: str = DEFAULT_HIDDEN_PATH) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Load pre-compiled hidden testcases."""
    if not os.path.exists(path):
        # Fallback to extract from dataset/train/train_ground_truth.tsv if not yet created
        alt_gt = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train", "train_ground_truth.tsv")
        if not os.path.exists(alt_gt):
            alt_gt = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")
        if os.path.exists(alt_gt):
            print(f"{YELLOW}[*] Compiling hidden testcases from {alt_gt}...{RESET}")
            # Extract first 30,000 entities
            ids = []
            gt_dict = {}
            with open(alt_gt, "r", encoding="utf-8") as f:
                next(f)
                for i, line in enumerate(f):
                    if i >= 30000:
                        break
                    s1, _, rest = line.partition(DELIM)
                    s1 = s1.strip()
                    ids.append(s1)
                    gt_dict[s1] = set(rest.strip().split(",")) if rest.strip() else set()
            return ids, gt_dict
        else:
            raise FileNotFoundError(f"Hidden testcases not found at {path} or in dataset/train!")

    ids = []
    gt_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            s1, _, rest = line.partition(DELIM)
            s1 = s1.strip()
            if not s1:
                continue
            ids.append(s1)
            gt_dict[s1] = set(rest.strip().split(",")) if rest.strip() else set()
    return ids, gt_dict


class PortalValidator:
    """Simulates the Unstop portal validation and sanity checker."""

    @staticmethod
    def validate_submission_file(file_path: str) -> Tuple[bool, List[str], Dict[str, Set[str]], int]:
        """
        Validates the output file against all challenge constraints:
          1. File exists and valid UTF-8
          2. File size <= 512 MB
          3. Exact header: source1_entity_id\tmatched_entity_ids
          4. Tab-separated (rejects CSV)
          5. No self matches (S1- in matched IDs)
          6. Only S2- and S3- prefixes
          7. No intra-list duplicates
          8. No duplicate source1_entity_id rows
        Returns: (is_valid, error_list, mapping, empty_singleton_count)
        """
        errors = []
        mapping = {}
        empty_count = 0

        if not os.path.isfile(file_path):
            return False, [f"File not found: {file_path}"], {}, 0

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > MAX_PORTAL_SIZE_MB:
            errors.append(f"File size {size_mb:.2f} MB exceeds portal limit of {MAX_PORTAL_SIZE_MB} MB!")

        seen_s1 = set()
        dup_s1 = set()
        intra_dupes = set()
        self_matches = set()
        wrong_prefixes = set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                header = f.readline()
                if not header:
                    return False, ["File is empty!"], {}, 0

                if DELIM not in header and "," in header:
                    errors.append("File appears comma-separated (CSV). Portal strictly requires TAB-separated (.tsv)!")
                    return False, errors, {}, 0

                cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
                if cols != EXPECTED_HEADER:
                    errors.append(f"Invalid header {cols}. Expected exactly {EXPECTED_HEADER}")
                    return False, errors, {}, 0

                for line_num, line in enumerate(f, start=2):
                    if not line.strip():
                        continue
                    s1, tab, rest = line.partition(DELIM)
                    if not tab:
                        errors.append(f"Line {line_num}: Malformed row without tab delimiter: {line[:50]!r}")
                        if len(errors) > 10:
                            break
                        continue

                    s1 = s1.strip()
                    if s1 in seen_s1:
                        dup_s1.add(s1)
                    seen_s1.add(s1)

                    ids_str = rest.strip()
                    if not ids_str:
                        empty_count += 1
                        mapping[s1] = set()
                        continue

                    ids = ids_str.split(",")
                    if len(ids) != len(set(ids)):
                        intra_dupes.add(s1)

                    id_set = set(ids)
                    mapping[s1] = id_set

                    for mid in id_set:
                        if mid.startswith("S1-"):
                            self_matches.add(mid)
                        elif not mid.startswith(("S2-", "S3-")):
                            wrong_prefixes.add(mid)

        except UnicodeDecodeError:
            errors.append("File is not valid UTF-8 text (portal requires UTF-8 encoding).")
            return False, errors, {}, 0
        except Exception as e:
            errors.append(f"Read error: {str(e)}")
            return False, errors, {}, 0

        if dup_s1:
            sample = list(dup_s1)[:3]
            errors.append(f"Duplicate source1_entity_id rows found ({len(dup_s1)} occurrences, e.g. {sample})")
        if intra_dupes:
            sample = list(intra_dupes)[:3]
            errors.append(f"Duplicate IDs inside matched list ({len(intra_dupes)} rows, e.g. {sample})")
        if self_matches:
            sample = list(self_matches)[:3]
            errors.append(f"Self-matches detected! IDs contain S1- prefix ({len(self_matches)} invalid IDs, e.g. {sample})")
        if wrong_prefixes:
            sample = list(wrong_prefixes)[:3]
            errors.append(f"Invalid entity prefix detected! IDs must start with S2- or S3- ({len(wrong_prefixes)} IDs, e.g. {sample})")

        is_valid = (len(errors) == 0)
        return is_valid, errors, mapping, empty_count


def evaluate_predictions(preds: Dict[str, Set[str]], test_ids: List[str], ground_truth: Dict[str, Set[str]]) -> Dict:
    """
    Computes official Unstop Leaderboard metrics on the given test IDs:
      - Overall F0.5
      - Public Leaderboard F0.5 (first 50%)
      - Private Leaderboard F0.5 (second 50%)
      - Macro Precision & Recall
      - Singleton accuracy
    """
    scores_f05 = []
    precisions = []
    recalls = []

    true_singletons = 0
    correct_singletons = 0
    total_matches_pred = 0

    n_total = len(test_ids)
    if n_total == 0:
        return {}

    split_idx = n_total // 2
    public_f05 = []
    private_f05 = []

    for i, s1_id in enumerate(test_ids):
        true_ids = ground_truth.get(s1_id, set())
        pred_ids = preds.get(s1_id, set())

        total_matches_pred += len(pred_ids)
        if not true_ids:
            true_singletons += 1
            if not pred_ids:
                correct_singletons += 1

        f05, prec, rec = compute_entity_f_beta(pred_ids, true_ids, beta=0.5)
        scores_f05.append(f05)
        precisions.append(prec)
        recalls.append(rec)

        if i < split_idx:
            public_f05.append(f05)
        else:
            private_f05.append(f05)

    macro_f05 = sum(scores_f05) / n_total
    macro_prec = sum(precisions) / n_total
    macro_rec = sum(recalls) / n_total

    pub_score = sum(public_f05) / len(public_f05) if public_f05 else 0.0
    priv_score = sum(private_f05) / len(private_f05) if private_f05 else 0.0
    singleton_acc = (correct_singletons / true_singletons) if true_singletons > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "public_f05": pub_score,
        "private_f05": priv_score,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_accuracy": singleton_acc,
        "total_matches_pred": total_matches_pred,
        "evaluated_count": n_total,
    }


def find_candidate_files() -> List[str]:
    """Discover all .tsv submission files across project folders."""
    search_dirs = [
        os.path.join(PROJECT_ROOT, "output", "submmision"),
        os.path.join(PROJECT_ROOT, "output", "submission"),
        os.path.join(PROJECT_ROOT, "output"),
        os.path.join(PROJECT_ROOT, "outputs"),
        os.path.join(PROJECT_ROOT, "Amazon-ML-Submission"),
    ]

    files = []
    seen = set()
    for d in search_dirs:
        if os.path.exists(d):
            for path in glob.glob(os.path.join(d, "*.tsv")):
                base = os.path.basename(path)
                # Ignore candidate_pairs files for scoring (they are candidate sets, not match results)
                if "candidate" in base.lower():
                    continue
                norm = os.path.abspath(path)
                if norm not in seen:
                    seen.add(norm)
                    files.append(norm)

    files.sort(key=lambda p: (not os.path.basename(p).startswith("matching_results_V"), os.path.basename(p)))
    return files


def run_benchmark(target_files: Optional[List[str]] = None, benchmark_size: int = 5000):
    """Main benchmark execution function."""
    print_banner()

    # 1. Load Hidden Testcases
    print(f"{BOLD}[Step 1/3] Loading Hidden Testcases Ground Truth...{RESET}")
    hidden_ids, ground_truth = load_hidden_ground_truth()
    print(f"  {GREEN}✔ Loaded {len(hidden_ids):,} hidden testcases from {os.path.relpath(DEFAULT_HIDDEN_PATH, PROJECT_ROOT)}{RESET}\n")

    # Benchmark subsets
    # Common 5K set (all models overlap here for 100% fair leaderboard comparison)
    common_5k = hidden_ids[:benchmark_size]
    common_5k_set = set(common_5k)

    # 2. Discover files
    if not target_files:
        target_files = find_candidate_files()

    print(f"{BOLD}[Step 2/3] Simulating Unstop Submission Portal on {len(target_files)} Output Files...{RESET}")
    print("=" * 95)

    results = []
    test_set_submissions = []

    for file_path in target_files:
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        base_name = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Validate with portal validator
        is_valid, errors, preds, empty_count = PortalValidator.validate_submission_file(file_path)

        if not is_valid:
            status = f"{RED}REJECTED ❌{RESET}"
            print(f"• {BOLD}{rel_path:<48s}{RESET} | Size: {file_size_mb:6.2f}MB | Status: {status}")
            for err in errors[:2]:
                print(f"    {RED}↳ Issue: {err}{RESET}")
            results.append({
                "file": rel_path,
                "base_name": base_name,
                "status": "REJECTED",
                "overall_f05": 0.0,
                "public_f05": 0.0,
                "private_f05": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "eval_count": 0,
                "total_rows": len(preds),
                "errors": errors,
            })
            continue

        # Check entity type: Hidden benchmark vs Full test set
        n_rows = len(preds)
        overlap_hidden = len(common_5k_set & set(preds.keys()))

        if overlap_hidden >= benchmark_size:
            # Evaluatable on Hidden Benchmark
            metrics = evaluate_predictions(preds, common_5k, ground_truth)
            status = f"{GREEN}SCORED 🏆{RESET}"

            # Also check if it covers 10K or 30K
            extended_info = ""
            if len(preds) >= 10000 and all(eid in preds for eid in hidden_ids[:10000]):
                m10 = evaluate_predictions(preds, hidden_ids[:10000], ground_truth)
                extended_info = f" (10K F0.5: {m10['macro_f05']:.5f})"

            print(
                f"• {BOLD}{rel_path:<48s}{RESET} | Status: {status} | "
                f"Overall F0.5: {BOLD}{CYAN}{metrics['macro_f05']:.5f}{RESET}{extended_info}"
            )

            results.append({
                "file": rel_path,
                "base_name": base_name,
                "status": "SCORED",
                "overall_f05": metrics["macro_f05"],
                "public_f05": metrics["public_f05"],
                "private_f05": metrics["private_f05"],
                "precision": metrics["macro_precision"],
                "recall": metrics["macro_recall"],
                "singleton_acc": metrics["singleton_accuracy"],
                "eval_count": benchmark_size,
                "total_rows": n_rows,
                "file_size_mb": file_size_mb,
            })

        else:
            # Official Test Set Submission (1.73M entities or partial test set)
            status = f"{BLUE}TEST SET (READY){RESET}"
            match_pct = ((n_rows - empty_count) / n_rows * 100) if n_rows > 0 else 0.0
            print(
                f"• {BOLD}{rel_path:<48s}{RESET} | Status: {status} | "
                f"Rows: {n_rows:,} | Match Rate: {match_pct:.1f}%"
            )
            test_set_submissions.append({
                "file": rel_path,
                "base_name": base_name,
                "total_rows": n_rows,
                "matches": n_rows - empty_count,
                "singletons": empty_count,
                "match_rate": match_pct,
                "file_size_mb": file_size_mb,
                "status": "PASSED VALIDATION",
            })

    # 3. Present Leaderboard
    print("\n" + "=" * 95)
    print(f"{BOLD}[Step 3/3] UNSTOP OFFICIAL LEADERBOARD RANKING (BENCHMARK: {benchmark_size:,} HIDDEN CASES){RESET}")
    print("=" * 95)

    scored_runs = [r for r in results if r["status"] == "SCORED"]
    scored_runs.sort(key=lambda x: x["overall_f05"], reverse=True)

    header = f"{'Rank':4s} | {'Submission Output File':44s} | {'Public (50%)':12s} | {'Private (50%)':13s} | {'Overall F0.5':12s} | {'Precision':9s} | {'Recall':8s}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * 115)

    medals = ["🥇 1", "🥈 2", "🥉 3"]
    for i, run in enumerate(scored_runs):
        rank_str = medals[i] if i < len(medals) else f"   {i+1}"
        row_str = (
            f"{rank_str:4s} | {run['file'][:44]:<44s} | "
            f"{run['public_f05']:12.5f} | {run['private_f05']:13.5f} | "
            f"{BOLD}{CYAN}{run['overall_f05']:12.5f}{RESET} | "
            f"{run['precision']:9.4f} | {run['recall']:8.4f}"
        )
        print(row_str)

    # 4. Highlight the Best Score File
    if scored_runs:
        best = scored_runs[0]
        print("\n" + "=" * 95)
        print(f"{BOLD}{YELLOW}🏆 BEST SCORING FILE IDENTIFIED:{RESET}")
        print(f"  {BOLD}File Path            :{RESET} {GREEN}{best['file']}{RESET}")
        print(f"  {BOLD}Overall F0.5 Score   :{RESET} {BOLD}{CYAN}{best['overall_f05']:.5f}{RESET} ({best['overall_f05']*100:.2f}%)")
        print(f"  {BOLD}Public Leaderboard   :{RESET} {best['public_f05']:.5f}")
        print(f"  {BOLD}Private Leaderboard  :{RESET} {best['private_f05']:.5f}")
        print(f"  {BOLD}Precision (2× Weight):{RESET} {best['precision']:.4f}")
        print(f"  {BOLD}Recall               :{RESET} {best['recall']:.4f}")
        print(f"  {BOLD}Singleton Precision  :{RESET} {best['singleton_acc']*100:.2f}%")
        print(f"  {BOLD}Portal Status        :{RESET} {GREEN}100% VALID & SCORED (Zero Submission Errors){RESET}")
        print("=" * 95)

    # 5. Full Competition Test-Set Submissions Summary
    if test_set_submissions:
        print(f"\n{BOLD}📋 FULL COMPETITION TEST SET SUBMISSIONS SUMMARY (dataset/test):{RESET}")
        print("-" * 95)
        for ts in test_set_submissions:
            print(
                f"  • {BOLD}{ts['file']:<45s}{RESET} | Rows: {ts['total_rows']:,} | "
                f"Matches: {ts['matches']:,} ({ts['match_rate']:.1f}%) | "
                f"Size: {ts['file_size_mb']:.1f}MB | {GREEN}Ready for Portal Upload ✔{RESET}"
            )
        print("-" * 95)

    # 6. Save JSON Leaderboard Report
    report_path = os.path.join(SCRIPT_DIR, "leaderboard_summary.json")
    report_data = {
        "benchmark_size": benchmark_size,
        "best_file": scored_runs[0] if scored_runs else None,
        "leaderboard": scored_runs,
        "test_set_submissions": test_set_submissions,
        "rejected_submissions": [r for r in results if r["status"] == "REJECTED"],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{GREEN}[+] Benchmark report saved to: {os.path.relpath(report_path, PROJECT_ROOT)}{RESET}\n")
    return scored_runs[0]["file"] if scored_runs else None


def main():
    parser = argparse.ArgumentParser(description="Unstop Portal Simulator & Submission Evaluator")
    parser.add_argument("--file", "-f", type=str, default=None, help="Evaluate a single output file")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Evaluate all .tsv files in a directory")
    parser.add_argument("--benchmark-size", "-n", type=int, default=5000, help="Number of hidden testcases to evaluate (default: 5000)")
    args = parser.parse_args()

    target_files = None
    if args.file:
        target_files = [args.file]
    elif args.dir:
        target_files = glob.glob(os.path.join(args.dir, "*.tsv"))

    run_benchmark(target_files=target_files, benchmark_size=args.benchmark_size)


if __name__ == "__main__":
    main()

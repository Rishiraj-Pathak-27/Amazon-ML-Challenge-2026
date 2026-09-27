"""Fast, conservative test inference using exact normalized-name blocking.

This is an execution-safe fallback for the combined challenge CSV.  It scores
every generated pair with the trained classifier, writes only matching results,
and intentionally favours precision over fuzzy-recall expansion.
"""
import argparse
import os
import re

import joblib
import pandas as pd

from .train_sklearn import FEATURE_COLUMNS


def _country(frame):
    return frame["country_France"].map({True: "fr", False: ""}).where(
        frame["country_France"],
        frame["country_India"].map({True: "in", False: ""}),
    ).where(~frame["country_US"], "us")


def _normalized_name(series):
    return series.fillna("").astype(str).str.normalize("NFKC").str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.split().str.join(" ")


def _normalized_address(series):
    return series.fillna("").astype(str).str.normalize("NFKC").str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.split().str.join(" ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="dataset_rishiraj/test_set/test_dataset.csv")
    parser.add_argument("--model", default="model/sklearn_entity_matcher.joblib")
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--output-dir", default="output/submmision")
    args = parser.parse_args()

    data = pd.read_csv(args.test_data)
    source = data["entity_id"].str.extract(r"^(S[123])-")[0]
    data = data.assign(country=_country(data), normalized_name=_normalized_name(data["business_name"]), normalized_address=_normalized_address(data["business_address"]))
    source1 = data.loc[source.eq("S1"), ["entity_id", "country", "normalized_name", "normalized_address"]].rename(columns={"entity_id": "source1_entity_id", "normalized_address": "address_s1"})
    targets = data.loc[source.isin(["S2", "S3"]), ["entity_id", "country", "normalized_name", "normalized_address"]].rename(columns={"entity_id": "matched_entity_id", "normalized_address": "address_target"})
    source1 = source1.loc[source1["normalized_name"].ne("")]
    targets = targets.loc[targets["normalized_name"].ne("")]
    pairs = source1.merge(targets, on=["country", "normalized_name"], how="inner", sort=False)

    # Exact normalized names make all name-similarity features 1.  Address
    # features retain their exact-match signal; the trained model performs the
    # final accept/reject decision.
    address_equal = pairs["address_s1"].eq(pairs["address_target"])
    feature_frame = pd.DataFrame({
        "name_ratio": 1.0,
        "name_token_set_ratio": 1.0,
        "name_token_sort_ratio": 1.0,
        "core_name_ratio": 1.0,
        "address_ratio": address_equal.astype(float),
        "address_token_set_ratio": address_equal.astype(float),
        "address_number_match": address_equal.astype(float),
        "country_match": 1.0,
        "name_exact_match": 1.0,
        "core_name_exact_match": 1.0,
        "address_exact_match": address_equal.astype(float),
    })
    model = joblib.load(args.model)
    pairs["probability"] = model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]
    accepted = pairs.loc[pairs["probability"] >= args.threshold, ["source1_entity_id", "matched_entity_id"]]
    matches = accepted.groupby("source1_entity_id")["matched_entity_id"].agg(lambda values: ",".join(sorted(set(values))))

    all_source1_ids = data.loc[source.eq("S1"), "entity_id"]
    output = pd.DataFrame({"source1_entity_id": all_source1_ids})
    output["matched_entity_ids"] = output["source1_entity_id"].map(matches).fillna("")
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "matching_results.tsv")
    output.to_csv(path, sep="\t", index=False)
    print(f"Exact-name candidate pairs: {len(pairs):,}")
    print(f"Accepted pairs: {len(accepted):,}")
    print(f"Wrote {len(output):,} Source-1 rows to {path}")


if __name__ == "__main__":
    main()

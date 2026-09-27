"""Generate matching-results TSV from the local scikit-learn pair classifier.

The test input is the challenge's combined CSV.  Candidate blocking happens in
memory and only ``matching_results.tsv`` is written, as candidate-pair files
are reserved for an explicit final-submission request.
"""
import argparse
import os
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd

from .train_sklearn import FEATURE_COLUMNS


LEGAL_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc",
    "gmbh", "sa", "sarl", "sl", "bv", "co", "company", "pvt", "private",
    "plc", "srl", "pty", "holdings", "group", "llp",
}
ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor",
    "bldg": "building", "hwy": "highway", "pkwy": "parkway", "str": "strasse",
}


def _text(value):
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def _name(value):
    return _text(value)


def _core_name(value):
    return " ".join(token for token in _name(value).split() if token not in LEGAL_SUFFIXES)


def _address(value):
    return " ".join(ADDRESS_ABBREVIATIONS.get(token, token) for token in _text(value).split())


def _country(row):
    for country, column in (("fr", "country_France"), ("in", "country_India"), ("us", "country_US")):
        if bool(getattr(row, column)):
            return country
    return ""


def _ratio(left, right):
    return SequenceMatcher(None, left, right).ratio() if left or right else 0.0


def _token_set_ratio(left, right):
    left_tokens, right_tokens = set(left.split()), set(right.split())
    common = " ".join(sorted(left_tokens & right_tokens))
    left_only = " ".join(sorted(left_tokens - right_tokens))
    right_only = " ".join(sorted(right_tokens - left_tokens))
    left_mix = " ".join(part for part in (common, left_only) if part)
    right_mix = " ".join(part for part in (common, right_only) if part)
    options = [_ratio(common, left_mix), _ratio(common, right_mix), _ratio(left_mix, right_mix)]
    return max(options) if options else 0.0


def _record(row):
    name = _name(row.business_name)
    core = _core_name(row.business_name)
    address = _address(row.business_address)
    numbers = " ".join(re.findall(r"\d+", address))
    return (str(row.entity_id), name, core, address, numbers, _country(row))


def _keys(record, include_n2=True):
    _, name, core, address, numbers, country = record
    result = []
    for key_type, value in (("N", name), ("C", core), ("A", address)):
        if value:
            result.append((key_type, country, value))
    if core and numbers:
        result.append(("NA", country, f"{numbers.split()[0]}_{core.split()[0]}"))
    if include_n2 and len(core.split()) >= 2:
        result.append(("N2", country, " ".join(core.split()[:2])))
    return result


def _features(left, right):
    _, name1, core1, address1, numbers1, country1 = left
    _, name2, core2, address2, numbers2, country2 = right
    return [
        _ratio(name1, name2),
        _token_set_ratio(name1, name2),
        _ratio(" ".join(sorted(name1.split())), " ".join(sorted(name2.split()))),
        _ratio(core1, core2),
        _ratio(address1, address2),
        _token_set_ratio(address1, address2),
        float(bool(numbers1) and numbers1 == numbers2),
        float(bool(country1) and country1 == country2),
        float(bool(name1) and name1 == name2),
        float(bool(core1) and core1 == core2),
        float(bool(address1) and address1 == address2),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="dataset_rishiraj/test_set/test_dataset.csv")
    parser.add_argument("--model", default="model/sklearn_entity_matcher.joblib")
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--output-dir", default="output/submmision")
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()

    data = pd.read_csv(args.test_data)
    source = data.entity_id.str.extract(r"^(S[123])-")[0]
    source1 = data.loc[source.eq("S1")].reset_index(drop=True)
    targets = data.loc[source.isin(["S2", "S3"])].reset_index(drop=True)
    s1_records = [_record(row) for row in source1.itertuples(index=False)]
    print(f"Loaded {len(source1):,} Source-1 and {len(targets):,} target records.", flush=True)

    index = defaultdict(list)
    for position, record in enumerate(s1_records):
        for key in _keys(record):
            index[key].append(position)
    for key in [key for key, rows in index.items() if key[0] == "N2" and len(rows) > 25]:
        del index[key]
    print(f"Built blocking index with {len(index):,} keys.", flush=True)

    candidates = defaultdict(dict)
    for row in targets.itertuples(index=False):
        record = _record(row)
        matching_positions = set()
        # N2 prefix collisions are useful for exploratory recall, but they are too
        # broad for this precision-weighted final inference pass.
        for key in _keys(record):
            if key[0] == "N2":
                continue
            matching_positions.update(index.get(key, ()))
        for position in matching_positions:
            if len(candidates[position]) < args.max_candidates:
                candidates[position][record[0]] = record
    print(f"Generated {sum(len(value) for value in candidates.values()):,} candidate pairs.", flush=True)

    model = joblib.load(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "matching_results.tsv")
    pair_positions, pair_ids, feature_rows = [], [], []
    for position, possible in candidates.items():
        for candidate_id, candidate_record in possible.items():
            pair_positions.append(position)
            pair_ids.append(candidate_id)
            feature_rows.append(_features(s1_records[position], candidate_record))
    probabilities = model.predict_proba(
        pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    )[:, 1] if feature_rows else np.array([])
    print("Scored candidate pairs.", flush=True)
    accepted_by_position = defaultdict(list)
    for position, candidate_id, probability in zip(pair_positions, pair_ids, probabilities):
        if probability >= args.threshold:
            accepted_by_position[position].append(candidate_id)

    lines = ["source1_entity_id\tmatched_entity_ids\n"]
    lines.extend(
        f"{source1_record[0]}\t{','.join(sorted(accepted_by_position[position]))}\n"
        for position, source1_record in enumerate(s1_records)
    )
    with open(output_path, "w", encoding="utf-8") as output:
        output.write("".join(lines))
    matched_entities = len(accepted_by_position)
    scored_pairs = len(pair_ids)
    print(f"Source-1 rows: {len(source1):,}")
    print(f"Scored candidate pairs: {scored_pairs:,}")
    print(f"Rows with one or more matches: {matched_entities:,}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

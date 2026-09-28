"""
Official Macro-F0.5 evaluation metric implementation matching the challenge formula.
"""


def entity_f_beta(predicted_ids, true_ids, beta=0.5):
    """Score a single Source-1 entity's prediction against ground truth."""
    if not true_ids and not predicted_ids:
        return 1.0  # correctly predicted singleton
    if not predicted_ids or not true_ids:
        return 0.0  # false positive on singleton, or missed match
    tp = len(predicted_ids & true_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(predictions, ground_truth, beta=0.5):
    """
    predictions:   {source1_entity_id: set(matched_ids)}
    ground_truth:  {source1_entity_id: set(true_matched_ids)}
    Averaged over every entity in ground_truth (singletons included).
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(entity_f_beta(pred_ids, true_ids, beta=beta))
    return sum(scores) / len(scores) if scores else 0.0

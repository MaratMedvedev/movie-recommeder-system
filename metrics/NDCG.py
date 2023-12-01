import math


def ndcg_metric(pred_scores, rel_scores):
    dcg_sc = 0
    for i, rel in enumerate(pred_scores):
        dcg_sc += (2 ** rel - 1) / (math.log2(i + 2))
    rel_scores.sort(reverse=True)
    ideal_dcg_sc = 0
    for i, rel in enumerate(rel_scores):
        ideal_dcg_sc += (2 ** rel - 1) / (math.log2(i + 2))
    return dcg_sc / ideal_dcg_sc

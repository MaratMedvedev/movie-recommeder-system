def average_f1_score(predictions, true_labels):
    """
    Calculate F1-score for each row of predicted item IDs and true item IDs up to the top-K.

    Parameters:
    - predictions: 2D array, each row represents predicted item IDs for a user.
    - true_labels: 2D array, each row represents true item IDs for a user.

    Returns:
    - Average of F1-scores
    """
    f1_scores = []
    for pred_row, true_row in zip(predictions, true_labels):
        common_items = set(pred_row).intersection(set(true_row))
        f1 = 2 * len(common_items) / (len(pred_row) + len(true_row))
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)

# predictions = [[1, 3, 5, 7], [2, 4, 6, 8]]
# true_labels = [[1, 2, 5, 7], [2, 4, 6, 8]]
# K = 2
#
# f1_scores = average_f1_score(predictions, true_labels)
# print(f"F1-scores for each row: {f1_scores}")

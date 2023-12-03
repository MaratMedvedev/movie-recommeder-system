import math


def calc_avg_RMSE(X, y, model):
    avg_RMSE = 0
    for i, user_preference in enumerate(X):
        RMSE = 0
        pred = model.predict(user_preference)
        for r, itemID in y[i]:
            RMSE += (pred[itemID] - r) ** 2
        RMSE = math.sqrt(RMSE / len(y[i]))
        avg_RMSE += RMSE
    avg_RMSE /= len(X)
    return avg_RMSE

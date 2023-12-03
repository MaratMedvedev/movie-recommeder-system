import math
import random
from models.CF_using_cosine_similarity import ItemToItemCollaborativeFiltering
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
rating_df = pd.read_csv("../data/raw/ml-100k/u.data", sep="\t", index_col=False, names=["userID", "itemID", "rating"])
user_items_list = []  # Will contains the list of items that user rate and ratings
for i in range(943):
    user_items_list.append([])

for index, r in rating_df.iterrows():
    user_items_list[r["userID"] - 1].append((r["rating"], r["itemID"] - 1))

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

def train_test_split(user_items_list, frac, num_of_items_that_user_rate):
    splt_idx = int(frac * len(user_items_list))
    train_data = user_items_list[:splt_idx]
    y_test = user_items_list[splt_idx:]
    X_test = []
    for items_rating in y_test:
        X_test.append(random.choices(items_rating, k=num_of_items_that_user_rate))
    return train_data, X_test, y_test

# Next, we will benchmark the model for different number of neighbours
# and different number of rated items for user.
# Of course, we expect that an increasing in the number of rated items will result
# in an increase in the quality of recommendations.

# Note: code below work about ≈50 minutes. If you want to evaluate model faster,
# you can drop some number of rated items or number of neighbours or increase the frac.

frac = 0.9 # fraction of data that will use for training
nums_items_user_rate = [1, 2, 5, 10, 15] # Number of rated items for the user
for num_of_items_that_user_rate in nums_items_user_rate:
    train_data, X_test, y_test = train_test_split(user_items_list, frac, num_of_items_that_user_rate)
    avg_RMSEs = []
    
    # Number of neighbours
    nbs = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1682]
    for nb in nbs:
        print(f"{nb} neigbours consider...")
        model = ItemToItemCollaborativeFiltering(n_users=len(train_data), n_items=1682, n_neighbors=nb)
        model.fit(train_data)
        avg_RMSEs.append(calc_avg_RMSE(X_test, y_test, model))

    plt.plot(nbs, avg_RMSEs, label=f'User rate {num_of_items_that_user_rate} items')

plt.xlabel('Number of neighbours')
plt.ylabel('RMSE')
plt.title('Dependence of model RMSE on neighbour number')
plt.legend()
plt.show()

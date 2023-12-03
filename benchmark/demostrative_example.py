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

model = ItemToItemCollaborativeFiltering(n_users=len(user_items_list), n_items=1682, n_neighbors=1682)
model.fit(user_items_list)

# For instance, I like very popular movies like this:
my_movies_ratings = [
    (5, 182),  # - Alien
    (5, 233),  # - Jaws
    (5, 97),   # - Silence of the Lambs
    (5, 10),   # - Se7en
    (5, 184),  # - Psycho
    (5, 199),  # - Shining
    (5, 178),  # - Clockwork Orange
    (5, 179),  # - Apocalypse Now
    (5, 194),  # - Terminator
    (5, 55),   # - Pulp Fiction
    (5, 187),  # - Full Metal Jacket
    (5, 99),   # - Fargo
]
# I think when you like movies similar to movies above you should not appreciate some not popular movie (1671) :)
pred_rating_for_not_popular_film = model.predict(my_movies_ratings)[800]
print("Prediction rating for some not popular film", pred_rating_for_not_popular_film)

# But for 'The  Godfather' it should pretty high
pred_rating_for_Godfather = model.predict(my_movies_ratings)[126]
print("Prediction rating for 'The Godfather'", pred_rating_for_Godfather)





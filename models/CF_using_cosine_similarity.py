from sklearn.neighbors import NearestNeighbors
import numpy as np

class ItemToItemCollaborativeFiltering:
    def __init__(self, n_items, n_users, n_neighbors=5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_neighbors = n_neighbors
        self.KNN = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')

    @staticmethod
    def mean(lst):
        if len(lst) == 0:
            return 0.
        return sum(lst) / len(lst)

    def fit(self, user_items_list):
        # usr_items_list should be in following format:
        # [
        # [(some_rating_of_user_0, some_item_idx_for_user_0), ...],
        # [(some_rating_of_user_1, some_item_idx_for_user_1), ...]
        # ]
        print("Main model preparation...")
        self.user_items_list = user_items_list  # Store list item rating for user
        self.item_users_list = []  # Store list user ratings for item
        for i in range(self.n_items):
            self.item_users_list.append([])

        for userID, items_ratings in enumerate(user_items_list):
            for rating, itemID in items_ratings:
                self.item_users_list[itemID].append((rating, userID))

        self.mtr = np.zeros((self.n_items, self.n_users), dtype=float)
        self.mu = 0
        self.n_ratings = 0
        for userID, itemRatings in enumerate(user_items_list):
            mean_rating = self.mean([i[0] for i in itemRatings])
            for rating, itemID in itemRatings:
                self.mu += rating
                self.n_ratings += 1
                self.mtr[itemID][userID] = rating - mean_rating

        self.mu /= self.n_ratings
        self.mean_items_rating = np.empty(self.n_items)
        for i in range(self.n_items):
            mean_item_i_rating = self.mean([p[0] for p in self.item_users_list[i]])
            self.mean_items_rating[i] = mean_item_i_rating
        self.KNN.fit(self.mtr)
        self.precalc_neighbors()

    def precalc_neighbors(self):
        print("Precalculating neighbors...")
        self.itemID_neighbors = []
        for itemID in range(self.n_items):
            distances, idxs = self.KNN.kneighbors([self.mtr[itemID]])
            item_distances = {}
            for i in range(self.n_neighbors):
                idx = idxs[0][i]
                distance = distances[0][i]
                item_distances[idx] = distance
            self.itemID_neighbors.append(item_distances)

    def predict_item_rating_using_bl_est(self, itemID, N, items_distances, items_ratings):
        new_mu = self.mu * self.n_ratings + sum(items_ratings.values())
        new_mu = new_mu / (self.n_ratings + len(items_ratings))
        user_mean_rating = self.mean(items_ratings.values())
        item_rating = items_ratings[itemID] if itemID in items_ratings else 0

        item_mean_rating = ((self.mean_items_rating[itemID] * len(self.item_users_list[itemID]) + item_rating) /
                            (len(self.item_users_list[itemID]) + 1))
        b_user_itemID = user_mean_rating + item_mean_rating - new_mu
        if not N:
            return b_user_itemID

        pred_user_rating_for_itemID = 0
        denominator = 0
        for cur_itemID in N:
            cur_item_mean_rating = self.mean_items_rating[cur_itemID] * len(self.item_users_list[cur_itemID]) + \
                                   items_ratings[cur_itemID]
            cur_item_mean_rating = cur_item_mean_rating / (len(self.item_users_list[cur_itemID]) + 1)
            b_user_cur_itemID = user_mean_rating + cur_item_mean_rating - new_mu

            pred_user_rating_for_itemID += items_distances[cur_itemID] * (items_ratings[cur_itemID] - b_user_cur_itemID)
            denominator += items_distances[cur_itemID]
        pred_user_rating_for_itemID /= denominator if denominator > 1e-10 else 1
        pred_user_rating_for_itemID += b_user_itemID
        return pred_user_rating_for_itemID

    def predict(self, userItems):
        # userItems should contain items that user already rated in the following format:
        # [(rating, item1), (rating, item2), ...]
        rating_prediction = np.empty(self.n_items)
        item_rating = {}
        for rating, itemID in userItems:
            item_rating[itemID] = rating

        for itemID in range(self.n_items):
            if itemID in item_rating:
                rating_prediction[itemID] = item_rating[itemID]
                continue
            idxs = self.itemID_neighbors[itemID].keys()
            item_distances = self.itemID_neighbors[itemID]
            # Items that given user rated and similar to item with "itemID" ID
            N = set(idxs) & set(item_rating.keys()) - {itemID}
            rating_prediction[itemID] = self.predict_item_rating_using_bl_est(itemID, N, item_distances, item_rating)

        return rating_prediction

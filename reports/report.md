# Introduction

In this report, I present an item-to-item collaborative filtering recommendation system based on centered cosine similarity (Pearson correlation).
Collaborative filtering is a popular technique for making personalized recommendations by leveraging user-item interactions.
The code implements a recommendation model using Nearest Neighbors with cosine similarity and incorporates a baseline estimation approach to predict user-item ratings.
# Data analysis

### Overview of the data
During the data overview, I used only `u.data`, `u.item`, `u.user`.
For `u.data` and `u.item` data, I get the films with the highest average rating:

![img.png](figures/thebestmovies.png)

During analyzing `u.user`, I calculate some statistics:
* **Occupation** distribution among the users:

![img.png](figures/occupation.png)

* **Gender** distribution among the users:

| Gender  | Num |
|---------|-----|
| Males   | 670 |
| Females | 273 |

* **Age** distribution among the users:

![img.png](figures/ages.png)

* And finally **geographic** distribution among the users:

![img.png](figures/geography_of_users.png)


### Data preprocessing

I do all data preprocessing in `movie_cosine_similarity.ipynb`. I use only `u.data` data.

Storing item-to-user matrix is not efficient method. So, I decide store data as list that contains the lists of items that user rate and ratings.

It looks like this:
~~~
item_to_user = [
                 [(some_rating_of_user_0, some_item_idx_for_user_0), ...],
                 [(some_rating_of_user_1, some_item_idx_for_user_1), ...],
                 ...
               ]
~~~
Index of this list represent some user and list that store at this index is user ratings.
These ratings just tuple pairs with user rating and item ID.  

After that I split users into two groups. One group I use to do prediction.

Other group I use to test my model. For each user from test group, I pick randomly **min_num_of_items_that_user_rate** ratings and just make left ratings zero.

*Note:* **min_num_of_items_that_user_rate** is variable from `movie_cosine_similarity.ipynb`

# Model Implementation

...

# Model Advantages and Disadvantages

**Advantages:**

1. Utilizes item-to-item collaborative filtering, capturing item similarities.
2. Incorporates baseline estimates for improved prediction rating.

**Disadvantages:**

1. Only use information about the user's preferences but not information about user themself.

# Training Process

...

# Evaluation

...

# Results

...
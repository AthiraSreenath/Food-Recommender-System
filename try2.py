import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Read data from CSV files
master_menu = pd.read_csv('data/master_menus.csv')
menu_item = pd.read_csv('data/master_menu_items.csv')
recipes = pd.read_csv('data/recipes.csv')
replacement_recipes = pd.read_csv('data/replacement_recipes.csv')

# Function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Calculate Jaccard similarity matrix
jaccard_matrix = pd.DataFrame(columns=menu_item["master_menu_item_seq"], index=menu_item["master_menu_item_seq"])
for i in range(len(jaccard_matrix)):
    for j in range(i, len(jaccard_matrix.columns)):
        if i == j:
            jaccard_matrix.loc[i, j] = 1
        else:
            menu_item1 = menu_item.loc[i]
            menu_item2 = menu_item.loc[j]
            recipe1 = recipes.loc[recipes["recipe_seq"] == menu_item1["default_recipe_seq"]]
            recipe2 = recipes.loc[recipes["recipe_seq"] == menu_item2["default_recipe_seq"]]
            ingredients1 = set(recipe1["assembly_instructions"].lower().split())
            ingredients2 = set(recipe2["assembly_instructions"].lower().split())
            jaccard_matrix.loc[i, j] = jaccard_matrix.loc[j, i] = jaccard_similarity(ingredients1, ingredients2)

# Cluster menus using K-Means
kmeans = KMeans(n_clusters=10)
kmeans.fit(jaccard_matrix)
menu_item["cluster"] = kmeans.labels_

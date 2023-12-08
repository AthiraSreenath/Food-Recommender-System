import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load Data and Preprocess

# Load CSV files
master_menu_df = pd.read_csv('data/master_menus.csv')
menu_item_df = pd.read_csv('data/master_menu_items.csv')
recipes_df = pd.read_csv('data/recipes.csv')
replacement_recipes_df = pd.read_csv('data/replacement_recipes.csv')

# Merge menu_item_df with recipes_df
merged_menu_recipes_df = pd.merge(menu_item_df, recipes_df, how='inner', left_on='default_recipe_seq', right_on='recipe_seq')


# Merge master_menu_df with merged_menu_recipes_df based on master_menu_seq
final_merged_df = pd.merge(master_menu_df, merged_menu_recipes_df, how='inner', on='master_menu_seq')

# Step 2: Jaccard Measure and Clustering

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Concatenate recipe_name and assembly_instructions for each recipe
recipe_descriptions = final_merged_df['recipe_name'] + ' ' + final_merged_df['assembly_instructions'].fillna('')

# Fit and transform the data
tfidf_matrix = vectorizer.fit_transform(recipe_descriptions)

# Convert TF-IDF matrix to a dense NumPy array
tfidf_matrix_dense = tfidf_matrix.toarray()

# Calculate Jaccard distances using dense array
jaccard_distances = pairwise_distances(tfidf_matrix_dense, metric='jaccard')


# Perform clustering (using k-means in this example)
k = 3  # Number of clusters 
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(jaccard_distances)

# Add cluster information to the merged dataframe
final_merged_df['cluster'] = clusters

# Step 3: Create a Knowledge Graph

# Create a graph
knowledge_graph = nx.Graph()

# Add nodes (recipes) and edges (substitutions)
for index, row in replacement_recipes_df.iterrows():
    knowledge_graph.add_edge(row['Recipe Id'], row['Replacement Recipe Id'])

# Visualize the graph (optional)
nx.draw(knowledge_graph, with_labels=True)
plt.show()

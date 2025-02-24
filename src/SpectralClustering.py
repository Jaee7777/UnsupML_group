#!/usr/bin/env python
# coding: utf-8

# # **Spectral Clustering for OnlineRetail and ShillBidding data** 

# **Data Pre-Processing** (Rerfer to 'OnlineRetail' file)

# In[51]:


import pandas as pd
retail = "../data/Online Retail.xlsx" # use this for using local data within the repo.
df_onlineRetail = pd.read_excel(retail, sheet_name='Online Retail')
df_onlineRetail.head()


# In[52]:


url = "../data/Shill Bidding Dataset.csv" # use local dataset within the repo instead.

# Load into a DataFrame
df_shillBidding = pd.read_csv(url)

# Convert Record_ID to object  
df_shillBidding['Record_ID'] = df_shillBidding['Record_ID'].astype(str)

# Convert Auction_ID to object  
df_shillBidding['Auction_ID'] = df_shillBidding['Auction_ID'].astype(str)

# Set Record_ID as index  
df_shillBidding.set_index('Record_ID', inplace=True)

# Remove Class variable from data - save for later use
target = df_shillBidding['Class']
df_shillBidding = df_shillBidding.drop(['Class'], axis=1)
df_shillBidding.head()


# In[53]:


print("df_onlineRetail data size", df_onlineRetail.shape)
print("df_shillBidding data size", df_shillBidding.shape)


# In[54]:


#Check dataframe for missing values
print("df_onlineRetail missing values: ", df_onlineRetail.isnull().sum())
print("df_shillBidding missing values: ", df_shillBidding.isnull().sum())


# **Drop missing values**

# In[55]:


# Drop missing Customer IDs and Description
df_onlineRetail.dropna(subset=["CustomerID", "Description"], inplace=True)
df_onlineRetail.shape


# In[56]:


df_onlineRetail["InvoiceDate"] = pd.to_datetime(df_onlineRetail["InvoiceDate"])
df_onlineRetail["InvoiceNo"] = df_onlineRetail["InvoiceNo"].astype(str)
df_onlineRetail.head()


# **Filtering Out Invalid Transactions for df_onlineRetail data**

# In[57]:


# Remove negative or zero values in Quantity and UnitPrice
df_onlineRetail = df_onlineRetail[(df_onlineRetail["Quantity"] > 0) & (df_onlineRetail["UnitPrice"] > 0)]


# ## **Create RFM Dataframe** (Rerfer to 'OnlineRetail' file)

# In[146]:


df_onlineRetaildf = df_onlineRetail.copy()

# Compute the latest transaction date in the dataset
latest_date = df_onlineRetail["InvoiceDate"].max()

# Create "TotalPrice" column manually using .loc to avoid warning
df_onlineRetail.loc[:, "TotalPrice"] = df_onlineRetail["Quantity"] * df_onlineRetail["UnitPrice"]

# Compute RFM metrics for each customer
rfm = df_onlineRetail.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (latest_date - x.max()).days),  # Days since last purchase
    Frequency=("InvoiceNo", "nunique"),  # Count of unique transactions
    Monetary=("TotalPrice", "sum")  # Total spending amount
).reset_index()  # Reset index

# Display the first five rows
rfm.head()

# Use MinMaxScaler for normalizing the original dataset to be used
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Apply log transformation (adding 1 to avoid log(0) issues)
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMaxScaler to Recency, Frequency, and Monetary
rfm_norm = scaler.fit_transform(rfm_log[['Recency', 'Frequency', 'Monetary']])

# Convert the result back into a DataFrame
rfm_minmax= pd.DataFrame(rfm_norm, columns=['Recency', 'Frequency', 'Monetary'])
rfm_minmax


# # *3D Clustering for onlineRetail Data*

# In[159]:


import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

# Copy the rfm_minmax data for 3d clustering
clustering_data_3D = copy.deepcopy(rfm_minmax)


# Visualize the data (3 features)
plt.style.use('seaborn-whitegrid')

# Create a figure and a 3D Axes
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using three PCA components
scatter = ax.scatter(clustering_data_3D['Recency'], clustering_data_3D['Frequency'], clustering_data_3D['Monetary'], s=5, edgecolor='k', color='blue')

# Labels and title
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
ax.set_title("3D Visualization of RFM features")

# Show the plot
plt.show()


# # Check Elbow Method best K

# In[148]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Initialize empty lists to store scores
wcss = []  # Within-Cluster Sum of Squares (Inertia)
silhouette_scores = []  # Silhouette Scores

# Define the range of cluster numbers to test (from 2 to 10 clusters)
K_range = range(2, 11)

# Iterate over different values of k
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Initialize K-Means
    kmeans.fit(rfm_minmax)  # Fit to MinMax-scaled RFM data
    
    # Append WCSS (inertia)
    wcss.append(kmeans.inertia_)
    
    # Compute silhouette score (only if k > 1)
    silhouette_avg = silhouette_score(rfm_minmax, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Create a dual-axis plot for both metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot WCSS (Elbow Method)
ax1.plot(K_range, wcss, marker='o', linestyle='--', color='blue', label="WCSS (Inertia)")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("WCSS (Inertia)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for silhouette scores
ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='red', label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and Legend
plt.title("Elbow Method & Silhouette Score for Optimal k")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1))

# Show the plot
plt.show()


# If we check inertia scores, best k might lie between 3 and 4 according to the elbow method. If we look at the silhouette score, k=4 will give relatively bigger score compared to k=3. So for SpectralClustering, k=4 is chosen. 

# In[164]:


from sklearn.cluster import SpectralClustering
cluster_results_3D = {}
# "Spectral Clustering"
algorithm = SpectralClustering(n_clusters=4, affinity='rbf', random_state=42)
labels = algorithm.fit_predict(clustering_data_3D)
cluster_results_3D["Spectral Clustering"] = labels

print(cluster_results_3D)
print(set(labels))


# **CLIQUE**

# In[185]:


from pyclustering.cluster.clique import clique

clique_instance = clique(clustering_data_3D.values.tolist(), 10, 2)
clique_instance.process()
clusters_clique = clique_instance.get_clusters()
labels = np.full(len(clustering_data_3D), -1)  # Initialize all as noise (-1)
for cluster_id, cluster in enumerate(clusters_clique):
    #print("cluster id, ",cluster_id, "cluster", cluster )
    for idx in cluster:
        labels[idx] = cluster_id 

cluster_results_3D["CLIQUE"] = labels

cluster_results_3D


# In[186]:


print(sum(labels == 2))
unique_labels = set(labels)
print("uniq",unique_labels)
print(len(labels))


# In[183]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
validation_scores = {}
# Compute validation metrics for each clustering method
for name, labels in cluster_results_3D.items():
    unique_labels = set(labels)
    # Remove noise from labels (i.e. CLIQUE)
    if -1 in unique_labels: 
        valid_mask = labels != -1  # Mask to filter out -1 values
        filtered_data = clustering_data_3D[valid_mask] 
        filtered_labels = labels[valid_mask]  # Keep only valid labels
        unique_labels.remove(-1)  # Exclude -1 from cluster count
    else:
        filtered_data = clustering_data_3D  # No need to filter
        filtered_labels = labels 



    if len(unique_labels) > 1:
        silhouette = silhouette_score(filtered_data, filtered_labels)
    else:
        silhouette = -1  # Assign -1 when only one cluster exists to avoid errors

    # Compute Davies-Bouldin Index (lower is better)
    db_index = davies_bouldin_score(clustering_data_3D, labels)

    # Compute Calinski-Harabasz Index (higher is better)
    ch_index = calinski_harabasz_score(clustering_data_3D, labels)

    # Store validation scores in a dictionary
    validation_scores[name] = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Index": ch_index
    }
print(validation_scores)


# In[187]:


# Set the style
plt.style.use('seaborn-whitegrid')

# Create a figure and a 3D Axes
fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(121, projection='3d')

clustering_data_3D['labels'] = cluster_results_3D['Spectral Clustering']

# Create a scatter plot for each cluster
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']  # Extend this list if more clusters

for i in range(cluster_results_3D['Spectral Clustering'].max() + 1):
    cur_cluster_data = clustering_data_3D[clustering_data_3D['labels'] == i]

    ax1.scatter(cur_cluster_data['Recency'], cur_cluster_data['Frequency'], cur_cluster_data['Monetary'], 
                s=2, color=colors[i], label=f'Cluster {i}')

# Labels and title
ax1.set_xlabel("Recency")
ax1.set_ylabel("Frequency")
ax1.set_zlabel("Monetary")
ax1.set_title("3D Visualization of RFM Features with Spectral Clustering")

# Legend
ax1.legend()


ax2 = fig.add_subplot(122, projection='3d')

clustering_data_3D['labels'] = cluster_results_3D['CLIQUE']
# Handle Noise Points (-1) in CLIQUE

noise_data = clustering_data_3D[clustering_data_3D['labels'] == -1]
if not noise_data.empty:
    ax2.scatter(noise_data['Recency'], noise_data['Frequency'], noise_data['Monetary'], 
                s=5, color='gray', label="Noise (-1)", alpha=0.5)

for i in range(cluster_results_3D['CLIQUE'].max() + 1):
    cur_cluster_data = clustering_data_3D[clustering_data_3D['labels'] == i]

    ax2.scatter(cur_cluster_data['Recency'], cur_cluster_data['Frequency'], cur_cluster_data['Monetary'], 
                s=2, color=colors[i], label=f'Cluster {i}')

# Labels and title
ax2.set_xlabel("Recency")
ax2.set_ylabel("Frequency")
ax2.set_zlabel("Monetary")
ax2.set_title("3D Visualization of RFM Features with CLIQUE")

# Legend
ax2.legend()


# Show the plot
plt.show()


# # *2 Dimensions After PCA*

# In[80]:


from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
# Apply PCA analysis on dataset that normalized using MinMaxscalar
# This would be used for all algorithm except DBSCAN
# Apply pca on datafram using 2 compopnents
pca_minmax = PCA(n_components=2)
rfm_reduced = pca_minmax.fit_transform(rfm_minmax)

# Convert the PCA output to a DataFrame
rfm_df = pd.DataFrame(rfm_reduced, columns=["PCA1", "PCA2"])

# Set the style
sns.set_style("whitegrid")

# Create a 2D scatter plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(rfm_df["PCA1"], rfm_df["PCA2"], s=50, edgecolor='k', color='blue')

# Labels and title
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("2D Visualization of RFM features After PCA")

# Show the plot
plt.show()


# In[81]:


from sklearn.cluster import SpectralClustering, DBSCAN, Birch
cluster_results_2D = {}
validation_scores = {}
# "Spectral Clustering"
algorithm = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
labels = algorithm.fit_predict(rfm_df)
cluster_results_2D["Spectral Clustering"] = labels


# In[82]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Compute validation metrics for each clustering method
for name, labels in cluster_results_2D.items():
    data = rfm_df
    if len(set(labels)) > 1:  # Silhouette Score requires at least 2 clusters
        silhouette = silhouette_score(data, labels)
    else:
        silhouette = -1  # Assign -1 when only one cluster exists to avoid errors

    # Compute Davies-Bouldin Index (lower is better)
    db_index = davies_bouldin_score(data, labels)

    # Compute Calinski-Harabasz Index (higher is better)
    ch_index = calinski_harabasz_score(data, labels)

    # Store validation scores in a dictionary
    validation_scores[name] = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Index": ch_index
    }
validation_scores
num_clusters = 1
fig, axes = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 5))
fig.suptitle("Comparison of Clustering Techniques on RFM Data", fontsize=14)
if num_clusters == 1:
    axes = [axes]

# Iterate through the clustering results and visualize the clusters
for i, (name, labels) in enumerate(cluster_results.items()):
    unique_labels = len(set(labels))  # Get number of unique clusters
    palette = sns.color_palette("tab10", n_colors=unique_labels)  # Define color palette

    # Scatter plot for visualizing clustering on Recency vs Frequency
    sns.scatterplot(
        x=rfm_df["PCA1"],  # X-axis: Recency
        y=rfm_df["PCA2"],  # Y-axis: Frequency
        hue=labels,  # Color clusters by labels
        palette=palette,  # Use color palette based on unique labels
        alpha=0.7,  # Set transparency for better visibility
        s=8,  # Set point size
        ax=axes[i]  # Assign plot to the respective subplot
    )

    # Set plot title and axis labels
    axes[i].set_title(f"{name}")
    axes[i].set_xlabel("Recency")
    axes[i].set_ylabel("Frequency")

    # Remove legends from each subplot to avoid redundancy
    axes[i].legend([], frameon=False)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Display the plots
plt.show()


# In[ ]:





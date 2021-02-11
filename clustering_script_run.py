

# Import packages +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from functions import load_from_mssql
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.manifold import TSNE


# Import data from SQL ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
str = "select * from playpen.katia_br_unsub_jm_final"
basedata = load_from_mssql(str)
print(basedata.head())


# Data preprocessing ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Remove the email column
basedata = basedata.drop(["email", "dcount_months", "dcount_projects"], axis=1)
# Change customer id to a text string
basedata["customer_id"] = basedata["customer_id"].astype("str")
# Create dependant variable and list of matrix of features
x = basedata.iloc[:, 1:].values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Using the elbow method to find the optimal number of clusters +++++++++++++++++++++++++++++++++++++++++++++++++
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Using the elbow method to find the optimal number of clusters +++++++++++++++++++++++++++++++++++++++++++++++++
# Set number of clusters from the elbow method
n = 5
# Perform the fit predict with the calcualted n of clusters
kmeans_final = KMeans(n_clusters = n, init = 'k-means++', random_state = 42)
clusters = kmeans_final.fit_predict(x_scaled)
# Combine clusters back with original data set
basedata["cluster"] = clusters.tolist()



# Visualising the clusters +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# TSNE method ---------------------------------------------------------------------
m = TSNE(learning_rate=50)
tsne_features = m.fit_transform(x_scaled)
basedata["x"] = tsne_features[:,0]
basedata["y"] = tsne_features[:,1]

sns.scatterplot(data = basedata, x = "x", y = "y", hue="cluster", palette="tab10")
plt.show()


# To_CSV ________________________________________________________________________________________________

basedata.to_excel("C:/Users/jonathan.manton/OneDrive - Immediate Media/Desktop/clustered_data.xlsx")




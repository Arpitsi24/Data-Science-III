#%%
# Arpit Singh
# B20084
# 6265104315

""" importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

df = pd.read_csv(r"Iris.csv")

# %% Question 1

# making species column and dropping column which are not required
species = []
for i in range (len(df['Species'])):
    if (df['Species'][i] == 'Iris-setosa'):
        species.append(0)
    if (df['Species'][i] == 'Iris-versicolor'):
        species.append(1)
    if (df['Species'][i] == 'Iris-virginica'):
        species.append(2)
df_to_pca = df.drop('Species' , axis=1)
attribute = df_to_pca.columns

# performing PCA and calculating the eigenvalues and eigenvectors and ploting the bar graph
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_to_pca)
eigenval,eigenvec= np.linalg.eig(np.cov(df_to_pca.T))
x = np.linspace(1,4,4)

plt.bar(x,[round(i,3) for i in eigenval], Color = "purple")
plt.title('Eigenvalue vs components')
plt.xlabel('eigenvalues')
plt.ylabel('no. of components')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.show()

#%% Question 2

# performing the Kmeans clustering on the reduced data and plotting the clusters
kmeans = KMeans(n_clusters=3)
b = kmeans.fit(df_pca)
label = kmeans.fit_predict(df_pca)
centers = kmeans.cluster_centers_

colour=['red','Black','orange']
label_names =['Iris-setosa','Iris-versicolor','Iris-virginica','centres']
for i in range(3):
    filtered_label = df_pca[label == i]
    plt.scatter(filtered_label[:,0], filtered_label[:,1],  color = colour[i])
plt.scatter([centers[i][0] for i in range (3)], [centers[i][1] for i in range (3)], color = 'Blue')
plt.legend(label_names)
plt.title(' K-means clustering for k=3')
plt.show()

# measuring distortion
print("distortion measure : ",b.inertia_)

#defining a function to calculate purity score
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

# measuring purity score
print("Purity score is : ", purity_score(species,label))
#%% Question 3

# performing the K-means clustering for different values of k and plotting the graphs
list=[]
pur_sco =[]
K = [2,3,4,5,6,7]
for i in K:
    kmeans = KMeans(n_clusters=i)
    b = kmeans.fit(df_pca)
    label = kmeans.fit_predict(df_pca)
    list.append(b.inertia_)
    pur_sco.append(purity_score(species , label))

plt.plot(K,list,color = "purple")
plt.title("Distortion measure vs Number of clusters")
plt.ylabel('no. of clusters')
plt.xlabel('distortion measure')
plt.show()
print(pur_sco)
# %% Question 4

# performing GMM clustering and plotting the graphs
gmm = GaussianMixture(n_components = 3)
gmm.fit(df_pca)
label = gmm.predict(df_pca)

colour=['red','Black','orange']
label_names =['Iris-setosa','Iris-versicolor','Iris-virginica','centres']
for i in range(3):
    filtered_label = df_pca[label == i]
    plt.scatter(filtered_label[:,0], filtered_label[:,1],  color = colour[i])
plt.scatter([centers[i][0] for i in range (3)] , [centers[i][1] for i in range (3)] , color = 'Blue')
plt.legend(label_names)
plt.title('gmm clustering for k = 3')
plt.show()

# calculating the value of likelihood
x = gmm.lower_bound_
print(x*(len(df_to_pca[attribute[1]])))
print(purity_score(species,label))

# %% Question 5

# performing the gmm for different no. of clusters
likelihood = []
pur_sco_gmm = []
K = [2,3,4,5,6,7]
for i in K:
    gmm = GaussianMixture(n_components = i , random_state=5)
    gmm.fit(df_pca)
    label = gmm.predict(df_pca)
    likelihood.append(gmm.lower_bound_ * len(df_to_pca[attribute[1]]))
    pur_sco_gmm.append(purity_score(species , label))
plt.plot(K,likelihood,color ="purple")
plt.show()
print(pur_sco_gmm)

# %% Question 6

#getting the species in the data frame for DBSCAN
def DBSCAN_(ep , samples):
    dbscan_model = DBSCAN(eps=ep, min_samples=samples).fit(df_pca)
    return dbscan_model.labels_

eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(df_pca)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for eps={eps[i]} min_samples={min_samples[i]} : ',round(purity_score(species, DBSCAN_predictions), 3))
    plt.scatter(df_pca[:,0], df_pca[:,1], c=DBSCAN_predictions, cmap='flag', s=15)
    plt.title(f'Data Points for eps={eps[i]} and min_samples={min_samples[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

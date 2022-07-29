#%%
# Arpit Singh
# B20084
# 6265104315
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment3\pima-indians-diabetes.csv")
att = ["pregs","plas","pres","skin","test","BMI","pedi","Age"]

#%% question 1

def outliers_(dataframe,attributes):           #finding outliers
    q1 = attributes.quantile([0.25][0])        #lower quartile
    q2 = attributes.quantile([0.5][0])         #median
    q3 = attributes.quantile([0.75][0])        #upper qualtile
    iqr = q3 - q1                              #interquartile range
    lower = q1 - 1.5*iqr                       #lower and upper limit values to find outliers
    upper = q3 + 1.5*iqr

    dfm = dataframe.copy()
    for i in range(len(attributes)):                                     #loop for finding the outliers which lies above or below the limit
        if lower > attributes.loc[i] or attributes.loc[i] > upper:
            attributes.loc[i] = q2                                       #filling outliers with meadian
        else:
            continue
    i = i + 1
    return()

dfol = df
for j in range(8):
    outliers_(dfol,dfol[att[j]])
    j = j+1

#question 1a

def normi(attributes):
    xn = np.max(attributes)
    yn = np.min(attributes)
    print('\nMin. value ',yn,'Max. value',xn)
    for i in range(len(attributes)):
        attributes.loc[i] = (attributes.loc[i]-yn) / (xn-yn)*(12-5) + 5
    print('Min. value ',np.min(attributes),'Max. value ',np.max(attributes))
    return()

dfnormi = dfol
for j in range(8):
    normi(dfnormi[att[j]])

#question 1b

def standi(attributes):
    xs = np.mean(attributes)
    ys = np.std(attributes)
    for i in range(len(attributes)):
        attributes.loc[i] = (attributes.loc[i]-xs)/ys
    return()

for k in range(8):
    x = att[k]
    print("\nmean of",att[k],df[x].mean())
    print("std deviation of",att[k],df[x].std())
    standi(df[x])
    print("\nmean of",att[k],"after standardization",df[x].mean())
    print("std deviation of",att[k],"after standardization",df[x].std())
    k = k+1
dfpca = df
dfpca=dfpca.drop(['class'],axis=1)

#%% question 2
# question 2a

mean = [0,0]
covariance = [[13,-3],[-3, 5]]
D = np.random.multivariate_normal(mean,covariance,1000,'ignore').T

plt.figure(figsize = (6,6))
plt.scatter(D[0],D[1],color ='purple')
plt.title('Scatter plot of 2D Synthetic data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim([-15,15])
plt.ylim([-15,15])
plt.show()

#question 2b

eigenvalue , eigenvectors = np.linalg.eig(np.cov(D))
print("\neigenvalues",eigenvalue)
print("eigenvectors",eigenvectors,"\n")
origin = [0,0]
eigen_vector1 = eigenvectors[:,0]
eigen_vector2 = eigenvectors[:,1]

plt.figure(figsize = (6,6))
plt.scatter(D[0],D[1],color ='purple')
plt.quiver(origin,origin,eigen_vector2,eigen_vector1,color ='red',scale = 5)
plt.title('Scatter plot of 2D Synthetic data and eigen directions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim([-15,15])
plt.ylim([-15,15])
plt.show()

#question 2c

e1 = np.dot(D.T,eigen_vector1)

plt.figure(figsize = (6,6))
plt.scatter(D[0],D[1],color ='purple')
plt.quiver(origin,origin,eigen_vector2,eigen_vector1,color ='red',scale =5)
plt.scatter(e1*eigenvectors[0][0],e1*eigenvectors[1][0],color ='orange')
plt.title('projected value onto 1st eigen directions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim([-15,15])
plt.ylim([-15,15])
plt.show()

e2 = np.dot(D.T,eigen_vector2)

plt.figure(figsize = (6,6))
plt.scatter(D[0],D[1],color ='purple')
plt.scatter(e2*eigenvectors[0][1],e2*eigenvectors[1][1],color ='orange')
plt.quiver(origin,origin,eigen_vector2,eigen_vector1,color ='red',scale = 5)
plt.title('projected value onto 2nd eigen directions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim([-15,15])
plt.ylim([-15,15])
plt.show()

#question 2d

D_ = np.dot(D.T,eigenvectors)
D_1 = np.dot(D_,eigenvectors.T)
MSE = np.square(np.subtract(D.T,D_1)).mean()
print(MSE)

#%% question 3
#question 3a
pca = PCA(n_components=2)
pca.fit(dfpca)
dfpca_tran = pca.transform(dfpca)

plt.scatter(dfpca_tran[:,0],dfpca_tran[:,1],color ='red')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Reduced dimensional data')
plt.show()

dfpca_same = dfpca
eigenvalues_,eigenvectors_ = np.linalg.eig(np.cov(dfpca_same.T))
u = pca.components_
v = pca.explained_variance_
print('Eigen value of dimension 1:',eigenvalues_[0])
print('Vairance of dimension 1:',v[0])
print('Eigen value of dimension 2:',eigenvalues_[1])
print('Variance of dimension 2:',v[1])

#question 3b
eigen_list = []
for i in eigenvalues_:
    eigen_list.append(i)
eigen_list.sort(reverse=True)

plt.bar(range(1,9),eigen_list,color ='red')
plt.xlabel('EigenValue')
plt.ylabel('magnitude')
plt.show()

#question 3c

def re(g,h):
    return (sum(((g-h)**2).sum(axis=1))/len(g))**0.5
dfpca_n = pca.inverse_transform(dfpca_tran)
print(re(dfpca,dfpca_n))
RE =[]

for i in range(1,8):
    pca = PCA(n_components=i)
    pca.fit(dfpca)
    dfpca_x = pca.transform(dfpca)
    dfpca_y = pca.inverse_transform(dfpca_x)
    RE.append(re(dfpca,dfpca_y))
print(RE)

plt.bar(range(1,8),RE,color ='red')
plt.plot(range(1,8),RE,color='green')
plt.scatter(range(1,8),RE)
plt.xlabel('No. of components')
plt.ylabel('Reconstruction error')
plt.show()

for n in range(1,9):
    pca1 = PCA(n_components=n)
    pcm = pca1.fit_transform(dfpca)
    cov_matrix = np.dot(pcm.T,pcm)
    print("\ncovariance matrix of l =",n,cov_matrix)

#question 3d

print("\ncovariance matrix for original data")
print(dfpca.cov())








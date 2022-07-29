#%%
# Arpit Singh
# B20084
# 6265104315
""" importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment4\SteelPlateFaults-2class.csv")

#%% q1

#split df according to class than into train and test according to the question
#creating csv files of train and test df
#finding the accuracy and confusion matrix of KNN classifier

data_split_by_class= df.groupby("Class")
dfclass_0 = data_split_by_class.get_group(0)
dfclass_1 = data_split_by_class.get_group(1)

[dfclass_0_train,dfclass_0_test,dfclass_0_label_train,dfclass_0_label_test] = train_test_split(dfclass_0,dfclass_0["Class"],test_size=0.3,random_state=42,shuffle=True)
[dfclass_1_train,dfclass_1_test,dfclass_1_label_train,dfclass_1_label_test] = train_test_split(dfclass_1,dfclass_1["Class"],test_size=0.3,random_state=42,shuffle=True)

df_train = pd.concat([dfclass_0_train,dfclass_1_train])
df_label_train = pd.concat([dfclass_0_label_train,dfclass_1_label_train])
df_train.to_csv("SteelPlateFaults-train.csv",index = False)

df_test = pd.concat([dfclass_0_test,dfclass_1_test])
df_label_test = pd.concat([dfclass_0_label_test,dfclass_1_label_test])
df_test.to_csv("SteelPlateFaults-test.csv",index = False)

k = [1,3,5]
for i in k:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(df_train,df_label_train)
    predict = KNN.predict(df_test)
    print("--------           --------           --------")
    print("for k = ",i)
    print("classification accuracy :- ",accuracy_score(df_label_test,predict),"\n")
    print("confusion matrix :- \n",confusion_matrix(df_label_test,predict))

#%% q2

#normalizing the df of test and train according to min-max value of train df
#creating csv files of normalized df
#finding the accuracy and confusion matrix of KNN classifier

df_train_to_normi = pd.read_csv('SteelPlateFaults-train.csv')
df_test_to_normi = pd.read_csv('SteelPlateFaults-test.csv')

max_dataset = df_train_to_normi.max()
min_dataset = df_train_to_normi.min()
dif_dataset = max_dataset-min_dataset
df_normi_train = pd.DataFrame()
df_normi_test = pd.DataFrame()
for v in df_train_to_normi:
    df_normi_train[v] = (df_train_to_normi[v] - min_dataset[v])/(dif_dataset[v])
for v in df_test_to_normi:
    df_normi_test[v] = (df_test_to_normi[v] - min_dataset[v])/(dif_dataset[v])

df_normi_train.to_csv("SteelPlateFaults-train-Normalised.csv", index=False)
df_normi_test.to_csv("SteelPlateFaults-test-normalised.csv", index=False)

df_train_new = df_normi_train.drop("Class",axis=1)
df_test_new = df_normi_test.drop("Class",axis=1)

k = [1,3,5]
for i in k:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(df_train_new,df_label_train)
    predict = KNN.predict(df_test_new)
    print("--------           --------           --------")
    print("for k = ",i)
    print("classification accuracy :- ",accuracy_score(df_label_test,predict),"\n")
    print("confusion matrix :- \n",confusion_matrix(df_label_test,predict))
print("--------           --------           --------")

#%% q3

#dropping unwanted attributes and splitting df according to class
#finding mean vector and covariance matrix
#finding likelihood using the expression of multivariate Gaussian density
#finding the accuracy and confusion matrix from bayes classifier

dftrain = pd.read_csv('SteelPlateFaults-train.csv')
dftest = pd.read_csv('SteelPlateFaults-test.csv')
dftest = dftest.drop(["Class","X_Minimum","Y_Minimum","TypeOfSteel_A300","TypeOfSteel_A400"],axis=1)
df_split_by_class= dftrain.groupby("Class")
dftrain_0 = df_split_by_class.get_group(0)
dftrain_0 = dftrain_0.drop(["Class","X_Minimum","Y_Minimum","TypeOfSteel_A300","TypeOfSteel_A400"],axis=1)
dftrain_1 = df_split_by_class.get_group(1)
dftrain_1 = dftrain_1.drop(["Class","X_Minimum","Y_Minimum","TypeOfSteel_A300","TypeOfSteel_A400"],axis=1)

cov0 = np.cov(dftrain_0.T)
cov1 = np.cov(dftrain_1.T)
mean0 = np.mean(dftrain_0)
mean1 = np.mean(dftrain_1)
print("mean of class 0 :-\n",mean0,"\n mean of class 1 :-\n",mean1)
cov0 = pd.DataFrame(cov0)
cov1 = pd.DataFrame(cov1)
cov0.to_csv("Covariancematrix_of_class0.csv", index=False)
cov1.to_csv("Covariancematrix_of_class1.csv", index=False)

def likelihood(xval, mvac, covmat):
    myMat = np.dot((xval-mvac).T, np.linalg.inv(covmat))
    inside = -0.5*np.dot(myMat, (xval-mvac))
    ex = np.exp(inside)
    return(ex/((2*np.pi)**5 * (np.linalg.det(covmat))**0.5))

p0 = len(dftrain_0)/len(dftrain)
p1 = len(dftrain_1)/len(dftrain)
predict_new = []
for i, row in dftest.iterrows():
    p_0 = likelihood(row, mean0, cov0) * p0
    p_1 = likelihood(row, mean1, cov1) * p1
    if p_0 > p_1:
        predict_new.append(0)
    else:
        predict_new.append(1)
print("--------           --------           --------")
print("Accuracy :- ", accuracy_score(df_label_test, predict_new),"\n")
print("Confusion Matrix :- \n", confusion_matrix(df_label_test, predict_new))
print("--------           --------           --------")
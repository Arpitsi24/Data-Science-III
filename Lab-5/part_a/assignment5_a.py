#%%
# Arpit Singh
# B20084
# 6265104315
""" importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment5\part_a\SteelPlateFaults-2class.csv')
df_test = pd.read_csv(r'C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment5\part_a\SteelPlateFaults-test.csv')
df_train = pd.read_csv(r'C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment5\part_a\SteelPlateFaults-train.csv')

#%%
# dropping attributes
df.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
df_test_c = df_test['Class']
df_test = df_test.drop(['Class'], axis= 1)

df_train = df_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
df_test = df_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
by_class = df_train.groupby('Class')
dfclass0_train = by_class.get_group(0)
dfclass1_train = by_class.get_group(1)
dfclass0_train = dfclass0_train.drop(['Class'], axis=1)
dfclass1_train = dfclass1_train.drop(['Class'], axis=1)

for q in [2,4,8,16]:
    GMM_class0 = GaussianMixture(n_components=q , covariance_type='full' , reg_covar=1e-4)
    GMM_class0.fit(dfclass0_train.values)
    GMM_class1 = GaussianMixture(n_components=q , covariance_type='full' , reg_covar=1e-4)
    GMM_class1.fit(dfclass1_train.values)

    prediction = []
    X = GMM_class0.score_samples(df_test.values)
    Y = GMM_class1.score_samples(df_test.values)
    for i in range(len(X)):
        if X[i] > Y[i]:
            prediction.append(0)
        if X[i] < Y[i]:
            prediction.append(1)

    conf_matrix = confusion_matrix (df_test_c.values, prediction)
    print("confusion matrix for q =",q,"is\n",conf_matrix)

    accuracy = accuracy_score(df_test_c.values, prediction)
    print("accuracy score for q =",q,"is :",accuracy)

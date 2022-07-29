#%%
# Arpit Singh
# B20084
# 6265104315
""" importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment5\part_b\abalone.csv")
#%%
#Q1
print("                       Q1\n")

#splitting train and test data and making csv file of train and test data
[abalone_train,abalone_test] = train_test_split(df,test_size=0.3,random_state=42,shuffle=True)
df_train = pd.DataFrame(abalone_train)
df_test = pd.DataFrame(abalone_test)
df_train.to_csv("abalone_train.csv",index = False)
df_test.to_csv("abalone_test.csv",index = False)

#finding pearson correlation coefficient of rings with other attributes
att = [df["Length"],df["Diameter"],df["Height"],df["Whole weight"],df["Shucked weight"],df["Viscera weight"],df["Shell weight"]]
attributes = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
corrmax = 0
for l in range(7):
    y = att[l]
    corr = df['Rings'].corr(y,method='pearson')
    if corr>corrmax:
        corrmax = corr
        corr_max_att = attributes[l]
    print("pearson correlation of "+ attributes[l]+" is      "+str(corr))
print(corr_max_att)  #Highest Pearson correlation coefficient attribute

#a simple linear (straight-line) regression model
x_dftrain = df_train.iloc[:,6].values.reshape(-1,1)
y_dftrain = df_train.iloc[:,7].values.reshape(-1,1)
x_dftest = df_test.iloc[:,6].values.reshape(-1,1)
y_dftest = df_test.iloc[:,7].values.reshape(-1,1)

def unilate_LR(x_given,y_given):
    reg = LinearRegression()
    reg.fit(x_given,y_given)
    y_pred = reg.predict(x_given)
    sum = 0
    for i in range(len(y_given)):
        sum = sum + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse = np.sqrt(sum/len(y_given))
    return float(E_rmse),y_pred

#prediction accuracy
print("Prediction Accuracy for train data: ",unilate_LR(x_dftrain,y_dftrain)[0])
print("Prediction Accuracy for test data: ",unilate_LR(x_dftest,y_dftest)[0])

#Plot the best fit line on the training data
plt.scatter(x_dftrain,y_dftrain,color='purple',label='Training Data')
plt.plot(x_dftrain,unilate_LR(x_dftrain,y_dftrain)[1],color='orange',label='Best Fit Line')
plt.title('Rings vs Shell weight best fit line on the training data')
plt.ylabel("Rings")
plt.xlabel(corr_max_att)
plt.legend()
plt.show()

#Plot the scatter plot of actual Rings(x-axis) vs predicted Rings(y-axis) on the test data
plt.scatter(y_dftest,unilate_LR(x_dftest,y_dftest)[1],color='purple',label='On Test data')
plt.title("Scatter plot of predicted rings vs. actual rings on test data")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.legend()
plt.show()
print("----------------------------------------------------------------------------------------------------------")

#%%
#Q2
print("                       Q2\n")

#a multivariate (multiple) linear regression model
x_dftrain=df_train.iloc[:,:-1].values
y_dftrain=df_train.iloc[:,7].values
x_dftest=df_test.iloc[:,:-1].values
y_dftest=df_test.iloc[:,7].values

def Multiple_LR(x_given,y_given):
    reg = LinearRegression()
    reg.fit(x_given,y_given)
    y_pred = reg.predict(x_given)
    sum=0
    for i in range(len(y_given)):
        sum=sum + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse=np.sqrt(sum/len(y_given))
    return float(E_rmse),y_pred

#Prediction Accuracy
print("Prediction Accuracy for Train data: ",Multiple_LR(x_dftrain,y_dftrain)[0])
print("Prediction Accuracy for Test data: ",Multiple_LR(x_dftest,y_dftest)[0])

#Plot  the  scatter  plot  of  actual Rings(x-axis)  vs  predicted Rings(y-axis)  on  the  test data
plt.scatter(y_dftest,Multiple_LR(x_dftest,y_dftest)[1],color='purple',label = 'On Test data')
plt.title("Scatter plot of predicted rings vs. actual rings on test data")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.legend()
plt.show()
print("----------------------------------------------------------------------------------------------------------")

#%%
#Q3
print("                       Q3\n")

#a simple nonlinear regression model using polynomial curve
x_dftrain = df_train.iloc[:,6].values.reshape(-1,1)
y_dftrain = df_train.iloc[:,7].values.reshape(-1,1)
x_dftest = df_test.iloc[:,6].values.reshape(-1,1)
y_dftest = df_test.iloc[:,7].values.reshape(-1,1)

def unilate_NLR(p,x_given,y_given):
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(x_dftrain)
    reg = LinearRegression()
    reg.fit(x_poly,y_dftrain)
    x_poly_given=poly_features.fit_transform(x_given)
    y_pred = reg.predict(x_poly_given)
    sum = 0
    for i in range(len(y_given)):
        sum = sum + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse=np.sqrt(sum/len(y_given))
    return float(E_rmse),y_pred

#prediction accuracy
print("Prediction accuracies of Train data")
p2 = unilate_NLR(2,x_dftrain,y_dftrain)[0]
p3 = unilate_NLR(3,x_dftrain,y_dftrain)[0]
p4 = unilate_NLR(4,x_dftrain,y_dftrain)[0]
p5 = unilate_NLR(5,x_dftrain,y_dftrain)[0]
for i in range(4):
    print("Prediction Accuracy for p = ",i+2,":",unilate_NLR(i+2,x_dftrain,y_dftrain)[0])
#Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis) for training data
plt.bar([2,3,4,5],[p2,p3,p4,p5],color='purple')
plt.ylim([2.42,2.51])
plt.title("Bar graph Rings vs Shell weight of training data")
plt.xlabel("degree of the polynomial")
plt.ylabel("Prediction Accuracy")
plt.show()

#prediction accuracy
print("Prediction accuracies of Test data")
p2 = unilate_NLR(2,x_dftest,y_dftest)[0]
p3 = unilate_NLR(3,x_dftest,y_dftest)[0]
p4 = unilate_NLR(4,x_dftest,y_dftest)[0]
p5,y5 = unilate_NLR(5,x_dftest,y_dftest)
for i in range(4):
    print("Prediction Accuracy for p = ",i+2,":",unilate_NLR(i+2,x_dftest,y_dftest)[0])
#Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis) for test data
plt.bar([2,3,4,5],[p2,p3,p4,p5],color='purple')
plt.ylim([2.38,2.44])
plt.title("Bar graph Rings vs Shell weight of test data")
plt.xlabel("degree of the polynomial")
plt.ylabel("Prediction Accuracy")
plt.show()

#Plot the best fit curve using the best fit model on the training data  where  the x-axis represents the shell weight and the y-axis is Rings.
plt.scatter(x_dftrain,y_dftrain,color='purple',label='Training Data')
plt.plot(x_dftest,y5,color='orange',label='Best Fit Line')
plt.title('Rings vs Shell weight best fit line on the training data')
plt.ylabel("Rings")
plt.xlabel("Shell weight")
plt.legend()
plt.show()

#Plot the scatter plot of the actual number of Rings(x-axis) vs the predicted number of Rings(y-axis) on the test data
p2,y_pred=unilate_NLR(2,x_dftest,y_dftest)
plt.scatter(y_dftest,y_pred,color='purple')
plt.title('Scatter plot of predicted rings vs. actual rings on test data')
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()
print("----------------------------------------------------------------------------------------------------------")

# %%
#Q4
print("                       Q4")

#a multivariate nonlinear regression model using polynomial regression
x_dftrain =df_train.iloc[:,:-1].values
y_dftrain = df_train.iloc[:,7].values
x_dftest = df_test.iloc[:,:-1].values
y_dftest = df_test.iloc[:,7].values

def Multivariate_NLR(p,x_given,y_given):
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(x_dftrain)
    reg = LinearRegression()
    reg.fit(x_poly,y_dftrain)
    x_poly_given=poly_features.fit_transform(x_given)
    y_pred = reg.predict(x_poly_given)
    sum = 0
    for i in range(len(y_given)):
        sum = sum + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse = np.sqrt(sum/len(y_given))
    return float(E_rmse),y_pred

#prediction accuracy
print("Prediction accuracies of Train data")
p2 = Multivariate_NLR(2,x_dftrain,y_dftrain)[0]
p3 = Multivariate_NLR(3,x_dftrain,y_dftrain)[0]
p4 = Multivariate_NLR(4,x_dftrain,y_dftrain)[0]
p5 = Multivariate_NLR(5,x_dftrain,y_dftrain)[0]
for i in range(4):
    print("Prediction Accuracy for p = ",i+2,":",Multivariate_NLR(i+2,x_dftrain,y_dftrain)[0])
#Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis) for training data
plt.bar([2,3,4,5],[p2,p3,p4,p5],color='purple')
plt.ylim([1.55,2.15])
plt.xlabel("degree of the polynomial")
plt.ylabel("Prediction Accuracy")
plt.show()

#prediction accuracy
print("Prediction accuracies of Test data")
p2 = Multivariate_NLR(2,x_dftest,y_dftest)[0]
p3 = Multivariate_NLR(3,x_dftest,y_dftest)[0]
p4 = Multivariate_NLR(4,x_dftest,y_dftest)[0]
p5 = Multivariate_NLR(5,x_dftest,y_dftest)[0]
for i in range(4):
    print("Prediction Accuracy for p = ",i+2,":",Multivariate_NLR(i+2,x_dftest,y_dftest)[0])
#Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis) for test data
plt.bar([2,3,4,5],[p2,p3,p4,p5],color='purple')
plt.xlabel("degree of the polynomial")
plt.ylabel("Prediction Accuracy")
plt.show()

#Plot the scatter plot of the actual number of Rings(x-axis) vs the predicted number of Rings(y-axis) on the test data
p2,y_pred=Multivariate_NLR(2,x_dftest,y_dftest)
plt.scatter(y_dftest,y_pred,color='purple')
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()
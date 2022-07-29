#%%
# Arpit Singh
# B20084
# 6265104315
""" importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.base import prediction
df = pd.read_csv(r"daily_covid_cases.csv")

#%%

# plotting the graph of covid cases per month
cases = df["new_cases"]
dates = df["Date"]
plt.plot(dates,cases)
xaxis = ["feb-20","apr-20","jun-20","aug-20","oct-20","dec-20","feb-21","apr-21","jun-21","aug-20","oct-21"]
plt.xticks([i for i in range(int(612/11),612,int(612/11))],xaxis)
plt.title('Plot of new covid cases per month')
plt.ylabel("New cases")
plt.xlabel("Month-Year")
plt.show()

# finding correlation coefficient using pearson method
df_lag1 = pd.DataFrame(df.iloc[1:])
df_skip1 = df.iloc[:611]
print(pearsonr(df_skip1["new_cases"],df_lag1["new_cases"])[0])

# plotting the graph of original and lag data for lag value 1
plt.scatter(df_skip1["new_cases"],df_lag1["new_cases"])
plt.title('Scatter plot of original and data with 1 day lag')
plt.ylabel("Original data")
plt.xlabel("Data with 1 day time lag")
plt.show()

# plotting diffrent graphs for lag value of 1 to 6
corr = []
lag = [1,2,3,4,5,6]
for i in lag:
    df_lag = pd.DataFrame(df.iloc[i:])
    df_skip = df.iloc[:612-i]
    corr.append(pearsonr(df_skip["new_cases"],df_lag["new_cases"])[0])
plt.plot(lag,corr)
plt.title('plot of correlation coefficient if original data with lag data')
plt.ylabel("correlation coefficient")
plt.xlabel("no. of lags")
plt.show()

plot_acf(df['new_cases'],lags = 6)
plt.yticks(np.arange(-1.25,1.5,step = 0.25))
plt.title('correlogram plot')
plt.xlabel("no. of lags")
plt.grid()
plt.show()

#%%

# dividing data into test and train
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_sz],X[len(X)-tst_sz:]

Window = 5 # The lag=5
model = AutoReg(train,lags=Window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print(coef)

history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# plotting test and prediction data for lag value 5
plt.scatter(test,predictions)
plt.title('scatter plot between actual and predicted values')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.show()

plt.plot(test)
plt.plot(predictions)
plt.show()

# finding rmse percentage and MAPE of the given test data
a = 0
b = len(test)
for i in range(b):
    a = a + (predictions[i]-test[i])**2
avg = sum(test)/len(test)
rmse = np.sqrt(a/len(test))
rmse_percent = (rmse/avg)*100
print(rmse_percent)

c=0
for i in range(b):
    c = c + abs(predictions[i]-test[i])/test[i]
mape = (c/b)*100
print(mape)

#%%

rmse_list = []
mape_list = []
lag_list = [1,5,10,15,25]
for iii in lag_list:
    Window = iii # The lag=5
    model = AutoReg(train,lags=Window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    history = train[len(train)-Window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-Window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(Window):
            yhat += coef[d+1] * lag[Window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    a = 0
    b = len(test)
    for iv in range(b):
        a = a + (predictions[iv]-test[iv])**2
    avg = sum(test)/len(test)
    rmse = np.sqrt(a/len(test))
    rmse_percent = float((rmse/avg)*100)
    rmse_list.append(rmse_percent)
    c=0
    for ivi in range(b):
        c = c + abs(predictions[ivi]-test[ivi])/test[ivi]
    mape = float((c/b)*100)
    mape_list.append(mape)
print(rmse_list,mape_list)

# plotting the rmse and MAPE values for diffrent values of lag
plt.bar(lag_list,rmse_list)
plt.xticks(lag_list)
plt.title('Scatter plot between actual and predicted values')
plt.xlabel('Lagged Value')
plt.ylabel('RMSE Value')
plt.show()

plt.bar(lag_list,mape_list)
plt.xticks(lag_list)
plt.title('Scatter plot between actual and predicted values')
plt.xlabel('Lagged Value')
plt.ylabel('MAPE Value')
plt.show()

# %%

# finding the heuristic value for the optimal number of lags
p = 1
flag = 1
while(flag == 1):
    newtrain = train[p:]
    l = len(newtrain)
    lag_newtrain = train[:l]
    epty = []
    eqty = []
    for i in range (len(newtrain)):
        epty.append(newtrain[i][0])
        eqty.append(lag_newtrain[i][0])
    corr = pearsonr(eqty,epty)
    if(2/math.sqrt(l)>abs(corr[0])):
        flag = 0
    else:
        p = p+1

print(p-1)
Window = p-1
model = AutoReg(train, lags=Window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# finding rmse percentage and MAPE of the given test data
a = 0
b = len(test)
for i in range(b):
    a = a + (predictions[i]-test[i])**2
avg = sum(test)/len(test)
rmse = np.sqrt(a/len(test))
rmse_percent = (rmse/avg)*100
print(rmse_percent)

c=0
for i in range(b):
    c = c + abs(predictions[i]-test[i])/test[i]
mape = (c/b)*100
print(mape)

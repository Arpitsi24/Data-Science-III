# Arpit Singh
# B20084
# 6265104315
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment2\landslide_data3_miss.csv")
dforiginal = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\New folder\assignment2\landslide_data3_original.csv")

#q1

missing_values = df.isnull().sum()          #adding the no. of missing values of attributes
attributes = list(missing_values.keys())    #attributes list
values = list(missing_values)               #missing values of attributes list
att = ['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']

plt.bar(attributes, values, width=0.4, color = "green")      #bar plot for plotting no. of missing values of all attributes
plt.xlabel('Attributes', fontsize = 15)
plt.title('No. of missing values vs attributes', fontsize = 15)
plt.show()

#q2 a)

drop_sid = df.dropna(subset = ["stationid"])     #dropping nan values in stationid attributes
after_drop_sid = len(drop_sid["dates"])          #length of data after dropping nan rows
before_drop_sid = 945
print("total no. of tuple deleted :",before_drop_sid - after_drop_sid)  #total no. of tuple deleted
#q2 b)

a = len(df.columns)
drop_onethird = df.dropna(thresh = (a - (a/3) + 1))    #dropping tuple having 3 or more nan values
after_drop_onethird = len(drop_onethird["dates"])      #length of tuple after deleting
before_drop_onethird = 926
print("total no. of tuple deleted :", before_drop_onethird - after_drop_onethird)   #number of tuple deleted in this step
drop_onethird = drop_onethird.drop(columns = ["dates"])         #creating some variables/dataframe for easy calculation
drop_onethird = drop_onethird.drop(columns = ["stationid"])
df_interpo = drop_onethird
df = drop_onethird
dfi = drop_onethird

#q3

missing_values_after = df.isnull().sum()              #finding missing values in each attributes
print(missing_values_after)
missing_values_after_df = df.isnull().sum().sum()
print("total no. of missing values after droping :",missing_values_after_df)

#q4 a)
print("------------------------------------------------------------------")

df_mean = df.fillna(df.mean())                      #filling nan values with mean of the attributes

for i in range(7):
    x = att[i]
    mean_miss = df[x].mean()                        #mean
    print('mean of',x,mean_miss)
    mode_miss = df[x].mode()                        #mode
    print('mode of',x,mode_miss)
    median_miss = df[x].median()                    #median
    print('median of',x,median_miss)
    std_miss = df[x].std()                          #standard deviation
    print('standard deviation of',x,std_miss)
    print('____________')
    i = i+1
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

list_of_att = [df['temperature'],df['humidity'],df['pressure'],df['rain'],df['lightavgw/o0'],df['lightmax'],df['moisture']]
list_att = ["temperature","humidity","pressure","rain","lightavgw/o0","lightmax","moisture"]
rmse_mean = []
for k in list_att:                                                      #loop for attributes
    sum = 0
    count = 0
    for ir in drop_onethird.index.values.tolist():                      #loop for every tuple
        if drop_onethird.isnull().at[ir,k] == True:                     #finding location of nan values
            sum = sum + ((df_mean.at[ir,k] - dforiginal.at[ir,k])**2)   #submission of difference of the real and replaced value for finding error
            count = count+1                                             #number of nan values
        else:
            continue
    rmse_mean.append(np.sqrt(sum/count))                                #making list of rmse values
plt.bar(list_att,rmse_mean,color="green")                               #bar plot of rmse values of the attributes
plt.show()

#i)
for i in range(7):
    x = att[i]
    mean = dforiginal[x].mean()                      #mean of original data
    print('mean of original',x,mean)
    mode = dforiginal[x].mode()                      #mode of original data
    print('mode of original',x,mode)
    median = dforiginal[x].median()                  #median of original data
    print('median of original',x,median)
    std = dforiginal[x].std()                        #standart deviation of original data
    print('standard deviation of original',x,std)
    print('____________')
    i = i+1
print("-----------------------------------------------------------------------------------------")

#q4 b)

df_inter = dfi.interpolate()                   #filling nan values with interpolate method

#i)
for i in range(7):
    x = att[i]
    mean_miss_i = dfi[x].mean()               #mean after interpolate method
    print('mean of',x,mean_miss_i)
    mode_miss_i = dfi[x].mode()               #mode after interpolate method
    print('mode of',x,mode_miss_i)
    median_miss_i = dfi[x].median()           #median after interpolate method
    print('median of',x,median_miss_i)
    std_miss_i = dfi[x].std()                 #standard deviation after interpolate method
    print('standard deviation of',x,std_miss_i)
    print('____________')
    i = i+1

rmse_inter = []
for k in list_att:                                                          #loop for attributes
    sum_ = 0
    count_ = 0
    for ir in df_interpo.index.values.tolist():                             #loop for every tuple
        if df_interpo.isnull().at[ir,k] == True:                            #finding location of nan values
            sum_ = sum_ + ((df_inter.at[ir,k] - dforiginal.at[ir,k])**2)    #submission of difference of the real and replaced value for finding error
            count_ = count_ + 1                                             #number of nan values
        else:
            continue
    if count_ == 0:
        rmse_inter = 0
    else:
        rmse_inter.append(np.sqrt(sum_/count_))                             #making list of rmse values
plt.bar(list_att,rmse_inter,color="green")                                  #bar plot of rmse values of the attributes
plt.show()
dfi.interpolate(inplace = True)                      #replacing values by interpolate method in the dataframe
#q5

def outliers_(dataframe, butes, inplace = True):     #finding outliers
    q1 = dataframe[butes].quantile([0.25][0])        #lower quartile
    q2 = dataframe[butes].quantile([0.5][0])         #median
    q3 = dataframe[butes].quantile([0.75][0])        #upper qualtile
    iqr = q3 - q1                                    #interquartile range
    lower = q1 - 1.5*iqr                             #lower and upper limit values to find outliers
    upper = q3 + 1.5*iqr

    outliers = []
    dfm = dataframe.copy()
    for i in dataframe[butes]:                      #loop for finding the outliers which lies above or below the limit
        if lower > i or i > upper:
            outliers.append(i)                      #list of values of outliers
            dfm.loc[dfm[butes] == i, butes] = q2    #filling outliers with meadian

    outset = set(outliers)
    if outset == set():
        return ["no outliers",dfm]
    else:
        return [outset,dfm]

def plot(dataframe, butes):               #boxplot
    plt.boxplot(dataframe[butes])
    plt.ylabel(butes)
    plt.show()

print("outliers for temperature :",outliers_(dfi,"temperature")[0])     #outliers of attribute temperature
print("outliers for rain :",outliers_(dfi,"rain")[0])                   #outliers of attribute rain
plot(dfi,"temperature")                                                 #boxplot of temperature attribute
plot(dfi,"rain")                                                        #boxplot of rain attribute
plot(outliers_(dfi,"temperature",inplace=False)[1],"temperature")       #boxplot after replacing outliers of temperature attribute with meadian
plot(outliers_(dfi,"rain",inplace=False)[1],"rain")                     #boxplot after replacing outliers of rain attribute with meadian

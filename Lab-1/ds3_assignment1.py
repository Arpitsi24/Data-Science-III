# Arpit Singh
# B20084
# 6265104315
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv(r"C:\Users\arpit\OneDrive\Documents\study material\ds3\pima-indians-diabetes.csv")
#q1
pregs = df['pregs']
plas = df['plas']
pres = df['pres']
skin = df['skin']
test = df['test']
bmi = df['BMI']
pedi = df['pedi']
age = df['Age']

attributes = [pregs,plas,pres,skin,test,bmi,pedi,age]
attributes_str = ['pregs','plas','pres','skin','test','bmi','pedi','age']

for i in range (8):
    x=attributes[i]

    mean = x.mean() #mean
    print("mean of "+attributes_str[i]+" "+str(mean))

    median = x.median() #median
    print("median of "+attributes_str[i]+" "+str(median))

    mode = x.mode() #mode
    print("mode of "+attributes_str[i]+" "+str(mode))

    minimum = x.min() #minimum
    print("minimum of "+attributes_str[i]+" "+str(minimum))

    maximum = x.max() #maximum
    print("maximum of "+attributes_str[i]+" "+str(maximum))

    std = x.std() #standard deviation
    print("standard deviation of "+attributes_str[i]+" "+str(std))

    print("-------------------------")

    i = i+1

#q2
att_age = ['pregs','plas','pres','skin','test','BMI','pedi']
for j in range(7):
    a = att_age[j]

    df.plot(kind='scatter',x='Age',y=a) #scatter plot between age and other attributes
    plt.xlabel('Age')
    plt.ylabel(a)
    plt.show()

    j = j+1

att_BMI = ['pregs','plas','pres','skin','test','pedi','Age']
for k in range(7):
    b = att_BMI[k]

    df.plot(kind='scatter',x='BMI',y=b) #scatter plot between BMI and other attributes
    plt.xlabel('BMI')
    plt.ylabel(b)
    plt.show()

    k = k+1


#q3
for l in range(8):
    y = attributes[l]

    corr1 = df['Age'].corr(y) #correlation coefficient of age with other attributes
    print("correlation coefficient of Age with "+ attributes_str[l]+" is "+str(corr1))

    corr2 = df['BMI'].corr(y) #correlation coefficient of BMI with other attributes
    print("correlation coefficient of BMI with "+ attributes_str[l]+" is "+str(corr2))

    l = l+1

#Q4
plt.hist(df['pregs']) #histogram of pregs
plt.xlabel('pregs')
plt.ylabel('frequency')
plt.show()

plt.hist(df['skin']) #histogram of skin
plt.xlabel('skin')
plt.ylabel('frequency')
plt.show()

#Q5
group = df.groupby('class') #histogram of pregs for each of the 2 classes
for o,p in group:
    plt.hist(p['pregs'])
    plt.xlabel('pregs')
    plt.ylabel('frequency')
    plt.show()

#Q6
for m in range(8):
    c = attributes[m]
    d = attributes_str[m]

    plt.boxplot(c) #boxplot of all attributes
    plt.ylabel(d)
    plt.show()
    m = m+1
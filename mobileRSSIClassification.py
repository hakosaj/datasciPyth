#Indoor-outdoor classification based on phone GSM-signal. Inspiration
#comes mostly from the following paper: 
#Indoor-Outdoor Detection Using a Smart Phone Sensor- Wang et al, 2016. MDPI.
#The GSM signals are used to determine if the phone (user) is
#in an indoor or outdoor environment. The data-analysis 
#and ML algorithms done mostly in Python using
#sklearn library.

#This project was part of Team Solaris' project in Junction 2019.
#Copyright: Jussi Hakosalo 2019


#Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statistics as st
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#General shit, paper etc

#The function to read in the data files measured on mobile devices.
#Input is expected to be an array of floats, each corresponding
#to the signal strength of a single cell station.
def openfile(fname,ind):

    with open (fname) as f:
        content=f.readlines()
    #Data formatting and cleaning
    content=[x.strip() for x in content]
    content=[x.strip("[") for x in content]
    content=[x.strip("]") for x in content]
    
    #These exist to make sure that we do not take identical consecutive measurements into account.
    #This is due to several details in the measurement application.
    previous=[]
    observation=[]

    for i in range(1,len(content)):
            observation=content[i].split(",")
            #Datasets used for training are either pure outdoor or
            #pure indoor measurement data. For that reason, we manually
            #assign binary labels to the data here.
            if ind:
                observation.insert(0,int(1))
            else:
                observation.insert(0,int(0))
            if previous!=observation:
                baseFeatures.append(list(map(lambda x:float(x),observation)))
            previous=content[i].split(",")
    

global baseFeatures
baseFeatures=[]

#Names for the datasets used in training the model
for name in ["indoors1_slow.txt","indoors_slow2.txt","indoors_slow3.txt"]:
    openfile(name,True)

for name in ["outdoors1.txt","outd2.txt"]:
    openfile(name,False)

#Data preprocessing and stuff here
baseFeatures.pop(0)
rar=pd.DataFrame.from_records(baseFeatures)
labels=rar[rar.columns[0]]
rar=rar.drop(rar.columns[0],axis=1)
rar[rar==0]=np.nan

#Transforming the measured values to a set of features to be used in training.
#We decided to use the following features, drawing ispiration from the afore-
#mentioned paper:
#Mean signal strength(abbrb. S), minimum SS, maximum SS, range of SS(max-min),
#standard deviation of SS, amount of cells available (indoors usually fewer
#cell stations reachable)

tempFeatures=[]
for index, row in rar.iterrows():
    vals = row.values.tolist()
    vals2=list(filter(lambda x: not math.isnan(x),vals))
    if(len(vals2)>1):
        tempFeatures.append([st.mean(vals2),min(vals2),max(vals2),max(vals2)-min(vals2),st.stdev(vals2),len(vals2)])
    else:
        vr=vals2[0]
        tempFeatures.append([vr,vr,vr,0,0,1])
features=pd.DataFrame.from_records(tempFeatures)

means=[]
stds=[]
for (columnName, columnData) in features.iteritems():
       rara=columnData.values
       means.append(st.mean(rara.tolist()))
       stds.append(st.stdev(rara.tolist()))
data_tuples=list(zip(means,stds))
stats=pd.DataFrame(data_tuples,columns=["Means","STDS"])
stats.to_csv("scaler.csv", encoding='utf-8', index=False)

#Splitting the data into train and test sets. Using standard deviation scaler
#to scale the features into more sensible range.

labels=labels.astype(int)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Code for both K-Nearest neighbors-classifier and Logistic Regression
#is below. Both were very accurate, with KNN being a few percentage
#points more accurate. However, since this classifier needs to be
#ported to Java, we use Logistic Regression there, since
#it is easy to implement in Android/Java environment.
#The model training is done here, and the weights of the
#classifier are then extracted and used later on.

print("Using KNN : \n")
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n Classification report")
print(classification_report(y_test, y_pred))
print("\n")
print(classifier)


print("Using Logistic Regression: \n")
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n Classification report")
print(classification_report(y_test,y_pred))
coefficients = pd.concat([pd.DataFrame(["Mean","Min","Max","Range","STD","nOfCells"]),pd.DataFrame(np.transpose(lr.coef_))], axis = 1)
print(coefficients)
coefficients.to_csv("coefs.csv", encoding='utf-8', index=False)
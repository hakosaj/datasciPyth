import pandas as pd
from sklearn import preprocessing

#survived class sex age siblings/spouses parents/children fare

X=pd.read_csv("train.csv")


#Drop unnecessary columns
dropThese = ['PassengerId','Name','Ticket','Cabin','Embarked']
X.drop(dropThese,inplace=True,axis=1)


#Enforce correct data types:
#for column in X.columns:
    #print(type(X.loc[2,column]))


#Sex to int. Men 0, female 1
gender={'male':0,'female':1}
X.Sex=[gender[item] for item in X.Sex]


print(X.shape)

#Drop all NA rows
X=X.dropna()
print(X.shape)

#Separate the columns
Y=X['Survived']
X.drop(['Survived'],inplace=True,axis=1)


#Calculate correlations between variables
#print(X.corr())

#Scale the columns
X=preprocessing.StandardScaler().fit(X).transform(X)
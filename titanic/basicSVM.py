from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#survived class sex age siblings/spouses parents/children fare

X=pd.read_csv("train.csv")

toPredict=pd.read_csv("test.csv")


#Drop unnecessary columns
dropThese = ['PassengerId','Name','Ticket','Cabin','Embarked']
X.drop(dropThese,inplace=True,axis=1)

ids=toPredict.PassengerId
dropThese=['PassengerId','Name','Ticket','Cabin','Embarked']
toPredict.drop(dropThese,inplace=True,axis=1)


#Enforce correct data types:
#for column in X.columns:
    #print(type(X.loc[2,column]))

print(toPredict.head())
#Sex to int. Men 0, female 1
gender={'male':0,'female':1}
X.Sex=[gender[item] for item in X.Sex]
toPredict.Sex=[gender[item] for item in toPredict.Sex]



#Drop all NA rows
X=X.dropna()
toPredict=toPredict.dropna()
print(X.shape)




#Separate the columns
Y=X['Survived']
X.drop(['Survived'],inplace=True,axis=1)


#Calculate correlations between variables
#print(X.corr())

#Scale the columns
X=preprocessing.StandardScaler().fit(X).transform(X)
toPredict=preprocessing.StandardScaler().fit(toPredict).transform(toPredict)






#Generate train set and test set
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)



#Choose model: Using support vector machine here
#clf = svm.SVC(gamma=0.001,C=100)
#Choose model: using logistic regression here
clf = LogisticRegression(random_state=0,solver='lbfgs')
#Choose model: using random forest here
#clf=RandomForestClassifier()


#Fit into model
clf.fit(X_train,y_train)

#Predict
pred=clf.predict(X_test)

#Print confusion matrix
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test,pred,labels=[1,0]))
print("\n")

#Print accuracy score
print("=== Accuracy Score ===")
print(accuracy_score(y_test,pred))
print("\n")

#Print  classification report
print("=== Classification Report ===")
print(classification_report(y_test, pred))
print('\n')

#Print cross validation scores
print("=== Cross validation scores ===")
cv_score = cross_val_score(clf, X, Y, cv=10, scoring='roc_auc')
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - LogReg: ", cv_score.mean())




predictions=clf.predict(toPredict)
print(predictions.shape)


print(predictions)
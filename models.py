from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression



#Generate train set and test set
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)


#Choose model: Using support vector machine here
#clf = svm.SVC(gamma=0.001,C=100)
#Choose model: using logistic regression here
#clf = LogisticRegression(random_state=0,solver='lbfgs')
#Choose model: using random forest here
#clf=RandomForestClassifier()


#Fit into model
clf.fit(X_train,y_train)

#Predict
pred=clf.predict(X_test)

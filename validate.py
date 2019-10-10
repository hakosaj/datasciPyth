
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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
print("Mean AUC Score - Random Forest: ", cv_score.mean())
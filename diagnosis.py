import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
dataset=pd.read_csv('wisc_bc_data.csv')

X=dataset.drop(['diagnosis','id'],axis=1)
y=dataset['diagnosis']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)
print(y_pred,y_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#confusion matrix
print(confusion_matrix(y_test,y_pred))

#accuracy score
print(accuracy_score(y_test,y_pred)*100)


plt.scatter(X_train,X_train,color="red")
plt.plot(X_train,clf.predict(X_train),"-",color='blue')
plt.title('salary vs experience (TEST SET)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

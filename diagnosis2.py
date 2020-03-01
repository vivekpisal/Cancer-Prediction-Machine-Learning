import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


ds=pd.read_csv('wisc_bc_data.csv')
X=ds.drop(['id','diagnosis'],axis=1)
y=ds['diagnosis']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)


from sklearn.svm import SVC

classify=SVC(kernel='linear',random_state=0)

classify.fit(X_train,y_train)

y_pred=classify.predict(X_test)

print(y_pred)

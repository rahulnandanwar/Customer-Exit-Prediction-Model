# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk

dataset = pd.read_csv("TrainingBank.csv")

dataset = pd.get_dummies(dataset, drop_first=True, columns=[ 'Geography','Gender'])

X = dataset.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1).values
y = dataset['Exited'].values

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=10)
X, y = sm.fit_sample(X, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Save the model on disk
pk.dump(sc, open('standard_scalar.pkl','wb'))

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 350, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)

print('Confusion Matrix for RF: \n',confusion_matrix(y_test, rf.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, rf.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, rf.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, rf.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, rf.predict(X_test)))

#Save the model on disk
pk.dump(rf, open('random_forest_model.pkl','wb'))


from fileinput import filename

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data=pd.read_csv("data/iris (1).csv")
#print(data.head())
print(data["species"].value_counts())

data["species"]=data["species"].map({
    "setosa":0,
    "versicolor":1,
    "virginica":2
})

print(data["species"].value_counts())

X=data.drop("species",axis=1)
y=data["species"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred)

accuracy=accuracy_score(y_test,y_pred)
print("model accuracy is:",accuracy)

joblib.dump(model,"model/model.pkl")
# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import standard libraries
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. import LogisticRegression from sklearn and apply the model on the dataset.
4. Predict the values of array
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S.SHAILESH KHANNA 
RegisterNumber: 212220040152 

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
HEAD:
![1](https://user-images.githubusercontent.com/103241457/171145451-cc65b610-d0ea-4446-8f03-5dc86cad9895.png)
Predicted value:
![2](https://user-images.githubusercontent.com/103241457/171145507-4200f8bf-7af8-4243-a337-66fa29e77582.png)
Accuracy:
![3](https://user-images.githubusercontent.com/103241457/171145569-f68d4f95-7a93-43f5-9b13-c4ed13b277c8.png)

Confusion Matrix:
![4](https://user-images.githubusercontent.com/103241457/171145614-42d4526c-daa0-4326-8b2e-37ca4c827825.png)

Classification Report:
![5](https://user-images.githubusercontent.com/103241457/171145671-08853da9-f785-412b-8982-be503f440a37.png)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

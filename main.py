import pandas as pd
import seaborn as srn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

diabetes_dataset=pd.read_csv("C:/Users/Venky/Desktop/D/diabetes.csv")
diabetes_dataset


diabetes_dataset.head() # print 1st 5 rows

srn.heatmap(diabetes_dataset.isnull())

diabetes_dataset.shape # print rows and cols

diabetes_dataset.describe() #statistical measures of dataset

diabetes_dataset['Outcome'].value_counts()

correlation=diabetes_dataset.corr()
print(correlation)

X=diabetes_dataset.drop(columns='Outcome',axis=1)
y=diabetes_dataset['Outcome']
print(X)


print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train

model=LogisticRegression()
model.fit(X_train,y_train)

prediction=model.predict(X_train)
prediction


prediction1=model.predict(X_test)
prediction1


accuracy=accuracy_score(prediction1,y_test)
print(accuracy)
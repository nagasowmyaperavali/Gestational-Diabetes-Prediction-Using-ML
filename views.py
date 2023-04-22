from django.shortcuts import render
import pandas as pd
import seaborn as srn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def home(request):
    return render(request, 'home.html')
def prediction(request):
    return render(request, 'prediction.html')
def result(request):
    diabetes_dataset = pd.read_csv(r"C:\Users\Venky\Desktop\D\diabetes.csv")
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    y = diabetes_dataset['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    val1=float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6= float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred=model.predict([[val1,val2,val3,val4,val5, val6,val7,val8]])
    result1 = ""
    if pred==[1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'prediction.html',{"result2":result1})


'''Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)'''

#%%
# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


df= pd.read_csv('C:/Users/LENOVO/Desktop/internship/diabetes.csv')
df.head()
df.shape
nullValues = df.isnull().sum().sum()#EDA : is to identify the pattterns through different data visualization
nullValues
duplicatedValues= df.duplicated().sum()
duplicatedValues

df.columns
df.info
df.describe
df.isnull().sum()

df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NAN)
p= df.hist(figsize=(12,10))
 
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(),inplace=True)
 
p= df.hist(figsize=(12,10))
p=msno.bar(df_copy)

'''
# one way to count values for our targer column here which is 'Out come'
there are two variables 0 for all who dont have the diabese , and 1 for who have

color_wheel = {1: "£0392cf", 3: "£7bc043"}
colors= df['Outcome'].map(lambda x: color_wheel.get(x+1))
print(df.Outcome.value_counts())
p=df.Outcome.value_counts().plot(kind='bar')

'''
# Used way will be as follow:
plt.title('Attrition Distribution')
df['Outcome'].value_counts().plot.pie(autopct='%1.2f%%')

#compare btw count of employees who have attririon and who'r not
categorical_count = df['Outcome'].value_counts().to_frame()
categorical_count # by value_counts function i breakdown the Attrition column into two values Yes and No to countvalues for each

sns.set_style('dark')
sns.countplot(x='Outcome',data=df) #Initial plot to compare btw yes and no who are left the company and who'r not. So we realized tht who were stayed more than who were left

plt.subplot(121), sns.distplot(df['Insulin'])
plt.subplot(122), df['Insulin'].plot.box(figsize=(16,4))

plt.figure(figsize=(12, 10))
p= sns.heatmap(df.corr(), annot=True)

df_copy.head()
scale = StandardScaler()
x= pd.DataFrame(scale.fit_transform(df_copy.drop(['Outcome'],axis=1),), columns=['Pregnancies','Insulin','Glucose', 'BloodPressure','SkinThickness','BMI','Age','DiabetesPedigreeFunction'])

# Split the data into features and target variable
X= df.drop('Outcome',axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=200)
from sklearn.metrics import accuracy_score,precision_score, recall_score
#Random Forest :
from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(n_estimators=200)
model1.fit(x_train, y_train)
model_tr_pred = model1.predict(x_train)
m1_tr=accuracy_score(model_tr_pred,y_train)
m1_tr #1.0
#prediction for test data
# model1.fit(x_test, y_test)
model1_tes_pred = model1.predict(x_test)
model1_tes= accuracy_score(model1_tes_pred, y_test)
model1_tes # 0.7

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, model1_tes_pred))
print(classification_report(y_test,model1_tes_pred))

#model2
from sklearn.tree import DecisionTreeClassifier
model2= DecisionTreeClassifier()
model2.fit(x_train,y_train)
model2_tra_pred=model2.predict(x_train)
model2_tra= accuracy_score(y_train,model2_tra_pred)
model2_tra # 1.0

#prediction for test data
model2_tes_pred = model2.predict(x_test)
model2_tes = accuracy_score(y_test,model2_tes_pred)
model2_tes# 0.67

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, model2_tes_pred))
print(classification_report(y_test,model2_tes_pred))

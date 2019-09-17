import nltk, os
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import svm
import pickle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv("/media/student/F0B618DAB618A360/Akash/Sem 2/Machine Learning/Module/Day 6/bank/bank.csv", sep = ';')

#print pd.get_dummies(df['marital'])
marital_dummy = []
education_dummy = []
default_dummy = []
job_dummy = []
housing_dummy = []
loan_dummy = []
contact_dummy = []
poutcome_dummy = []
y_dummy = []

cat_names = {'single':0, 'married':1, 'divorced':2}
for elem in df['marital']:
    marital_dummy.append(cat_names[elem])
df['marital_dummy'] = pd.Series(marital_dummy)  

cat_names2 = {'primary':0, 'secondary':1, 'tertiary':2, 'unknown':3}
for elem in df['education']:
    education_dummy.append(cat_names2[elem])
df['education_dummy'] = pd.Series(education_dummy)  

cat_names3 = {'no':0, 'yes':1}
for elem in df['default']:
    default_dummy.append(cat_names3[elem])
df['default_dummy'] = pd.Series(default_dummy)  

cat_job = {'unemployed':0, 'services':1, 'management':2, 'blue-collar':3, 'self-employed':4, 'technician':5, 'entrepreneur':6, 'admin.':7, 'student':8, 'housemaid':9, 'retired':10, 'unknown': 11}
for elem in df['job']:
    job_dummy.append(cat_job[elem])
df['job_dummy'] = pd.Series(job_dummy)  

cat_housing = {'no':0, 'yes':1}
for elem in df['housing']:
    housing_dummy.append(cat_housing[elem])
df['housing_dummy'] = pd.Series(housing_dummy)  

cat_loan = {'no':0, 'yes':1}
for elem in df['loan']:
    loan_dummy.append(cat_loan[elem])
df['loan_dummy'] = pd.Series(loan_dummy)

cat_contact = {'cellular':0, 'unknown':1, 'telephone':2}
for elem in df['contact']:
    contact_dummy.append(cat_contact[elem])
df['contact_dummy'] = pd.Series(contact_dummy)  

cat_poutcome = {'unknown':0, 'failure':1, 'other':2, 'success':3}
for elem in df['poutcome']:
    poutcome_dummy.append(cat_poutcome[elem])
df['poutcome_dummy'] = pd.Series(poutcome_dummy)

cat_y = {'no':0, 'yes':1}
for elem in df['y']:
    y_dummy.append(cat_y[elem])
df['y_dummy'] = pd.Series(y_dummy) 

print df['job_dummy'], df['marital_dummy'], df['education_dummy'], df['default_dummy'], df['housing_dummy'], df['loan_dummy'], df['contact_dummy'], df['poutcome_dummy'], df['y_dummy']


x = df.drop(['loan','day','month','duration'], axis = 1)
x = x.drop(['campaign','pdays','previous','y'], axis = 1)
x = x.drop(['job','marital','education','default'], axis = 1)
x = x.drop(['housing','loan_dummy','contact','poutcome'], axis = 1)
x = x.drop(['marital_dummy', 'education_dummy'], axis = 1)

y = df['loan']
y = pd.get_dummies(y)['yes']



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=123)
#model = LogisticRegression()
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print x

print 'ACCURACY SCORE: ', metrics.accuracy_score(y_test,pred)



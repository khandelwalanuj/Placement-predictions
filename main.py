import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df=pd.read_csv('Placement_Data_Full_Class.csv')

df.head()

df.set_index('sl_no')

df.info()

df.describe()

df.isna().sum()

df.drop(['salary'],axis=1,inplace=True)

df.columns

df.nunique()

sns.distplot(df['mba_p'])

sns.boxplot(x='ssc_b',y='ssc_p',data=df,hue='gender')

sns.boxplot(x='hsc_b',y='hsc_p',data=df,hue='gender')

sns.distplot(df['ssc_p'],label='ssc',kde=False)
sns.distplot(df['hsc_p'],label='hsc',kde=False)
plt.legend()
plt.title('percentage in hsc and ssc')
plt.xlabel('boards')



Gender=pd.get_dummies(df['gender'])
Ssc_b=pd.get_dummies(df['ssc_b'])
Hsc_b=pd.get_dummies(df['hsc_b'])
Hsc_s=pd.get_dummies(df['hsc_s'])
Degree_t=pd.get_dummies(df['degree_t'])
Workex=pd.get_dummies(df['workex'])
Specialisation=pd.get_dummies(df['specialisation'])

df.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'],axis=1,inplace=True)

df.head()

df=pd.concat([df,Gender,Ssc_b,Hsc_b,Hsc_s,Degree_t,Workex,Specialisation],axis=1)

df.head()

df['Central_ssc']=df.iloc[:,9:10]
df['Central_hsc']=df.iloc[:,11:12]

df.drop(df.iloc[:,9:10],axis=1,inplace=True)

df.columns

df.drop(['F','Others','Arts','No','Mkt&Fin'],axis=1,inplace=True)

df.head()

df.drop(['sl_no'],axis=1,inplace=True)

Status=pd.get_dummies(df['status'],drop_first=True)


df.drop(['status'],axis=1,inplace=True)
df=pd.concat([df,Status],axis=1)

df1=df.copy()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X=df1.drop(['Placed'],axis=1)
y=df1['Placed']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

log=LogisticRegression()
log.fit(X_train,y_train)
pred=log.predict(X_test)

print('Train Score: ', log.score(X_train,y_train))  
print('Test Score: ', log.score(X_test, y_test))  

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))



df2=df.copy()
sc1=StandardScaler()
sc1.fit(df2.drop('Placed',axis=1))
sc_feat=sc1.transform(df2.drop('Placed',axis=1))
df_scaled=pd.DataFrame(sc_feat,columns=df2.columns[:-1])

X_train1,X_test1,y_train1,y_test1=train_test_split(df_scaled,df2['Placed'],test_size=0.2,random_state=41)

error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train1,y_train1)
    pred_i=knn.predict(X_test1)
    error_rate.append(np.mean(pred_i != y_test1))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,ls='--')
plt.title('error rate vs K')
plt.xlabel('K')
plt.ylabel('Error rate')

knn1=KNeighborsClassifier(n_neighbors=24)
knn1.fit(X_train1,y_train1)
pred=knn1.predict(X_test1)

print(classification_report(y_test1,pred))

print('Train Score: ', knn1.score(X_train1,y_train1))  
print('Test Score: ', knn1.score(X_test1, y_test1))  
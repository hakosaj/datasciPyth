import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


#Get data
data=pd.read_csv('train.csv')

#Check the head
print(data.head())

#Check how many NA or NULL values
print("Null values around?")
print(data.isnull().sum())


#Plot: how many survived?

#f,ax=plt.subplots(1,2,figsize=(18,8))
#data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
#ax[0].set_title('Survived')
#ax[0].set_ylabel('')
#sns.countplot('Survived',data=data,ax=ax[1])
#ax[1].set_title('Survived')
#plt.show()  

#Categorical features: one or more categories, we cannot sort explicitly
#Ordinal features: Not continuous but can be sorted
# :Height(Short,Medium,Tall) (PClass)
#Continuous: Age



#Categorical feature: Sex
#Review values
#print(data.groupby(['Sex','Survived'])['Survived'].count())
#Plot values
#f,ax=plt.subplots(1,2,figsize=(7,3))
#data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
#ax[0].set_title('Survived vs Sex')
#sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
#ax[1].set_title('Sex:Survived vs Dead')
#plt.show()


#Ordinal feature PClass
#Plot in cross matrix and other features of the data

#print(pd.crosstab(data.Pclass,data.Survived,margins=True))
#f,ax=plt.subplots(1,2,figsize=(7,3))
#data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
#ax[0].set_title('Number Of Passengers By Pclass')
#ax[0].set_ylabel('Count')
#sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
#ax[1].set_title('Pclass:Survived vs Dead')
#plt.show()



#
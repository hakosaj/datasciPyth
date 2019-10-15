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
print(data.isnull().sum())


#Plot: how many survived?

sns.countplot('Survived',data=data)
plt.show()
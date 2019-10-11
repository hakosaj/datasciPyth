from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Lambda
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


train =pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

submission = pd.read_csv("sample_submission.csv")


#Scale the values in 0-1
X = train.iloc[:,1:]/255.
y = train.iloc[:,0]
test=test/255

#Reshape 784 to 28-bit squares
X = X.values.reshape(train.shape[0],28,28,1)
test = test.values.reshape(test.shape[0],28,28,1)

#Target values to Keras categorical
y = to_categorical(y)

#Split train and test
Xtrain,Xtest,ytrain,ytest= train_test_split(X,y,
test_size=0.15,random_state=29)

#Normalization
mean=Xtrain.mean().astype(np.float32)
std = Xtrain.std().astype(np.float32)
def standardize(x):
    return (x-mean)/std

#Model for the CNN
def CNN():
    model = models.Sequential()
    model.add(Lambda(standardize,input_shape=(28,28,1)))
    model.add(Convolution2D(32,(3,3),activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Convolution2D(128,(2,2),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0,4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0,3))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#Fit and boom
classifier = CNN()

classifier.fit(Xtrain,ytrain,epochs=20,batch_size=1000,validation_data=(Xtest,ytest))

#Predictions
prediction=classifier.predict(test)
prefictions=np.argmax(prediction,axis=1)

# submission
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("simple_cnn_kaggle.csv", index=False, header=True)


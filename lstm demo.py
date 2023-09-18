# this code reads images and feeds them into an lstm model for detection

import cv2
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#read training data
trainX = []

target_size = (1000,1000)
#edge
for i in [1,3,4,5]:
    img = cv2.imread("Training\\Edge\\e{0}.jpg".format(i))

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = np.reshape(img,(3000000))
    trainX.append(img)

# john cena
for i in range(1,6):
    img = cv2.imread("Training\\JohnCena\\j{0}.jpg".format(i))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = np.reshape(img,(3000000))
    trainX.append(img)

#Roman reigns
for i in range(1,6):
    img = cv2.imread("Training\\RomanReigns\\r{0}.jpg".format(i))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = np.reshape(img,(3000000))
    trainX.append(img)


#labels
labels = ["Edge","Edge","Edge","Edge","John Cena","John Cena","John Cena","John Cena","John Cena","Roman Reigns","Roman Reigns","Roman Reigns","Roman Reigns","Roman Reigns"]
labelsEncoded=[]
for label in labels:
    if label=="Edge":
        labelsEncoded.append(0)
    elif label=="John Cena":
        labelsEncoded.append(1)
    else:
        labelsEncoded.append(2)

#reshape input to [samples,timesteps,features]
#current format = [samples,features]
trainX = np.array(trainX)
trainX = np.resize(trainX,(14,1,3000000))

#create model
model = Sequential()
model.add(LSTM(4,input_shape=(1,3000000)))
model.add(Dense(1))

labelsEncoded = np.array(labelsEncoded)

print("X shape",trainX.shape)
print("y shape",labelsEncoded.shape)

#train model

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, labelsEncoded, epochs=100, batch_size=1, verbose=2)

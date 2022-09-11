import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

import keras
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
import tensorflow as tf
import csv
from timeit import default_timer as timer

from scipy.signal import argrelextrema

###Task 1
with np.load("data/train_data_label.npz") as data:
    train_data = data["train_data"]
    train_label = data["train_label"]

with np.load("data/test_data_label.npz") as data:
    test_data = data["test_data"]
    test_label = data["test_label"]

X_train, X_valid = train_test_split(train_data, test_size=0.2, random_state = 21)
y_train, y_valid = train_test_split(train_label, test_size=0.2, random_state = 21)

##KNN

#Hyperparameter tuning for KNN
scores = []
for i in range(1,11):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    score = accuracy_score(y_valid, preds)
    scores.append(score.mean())
print(scores)
# k=1 performs best with a validation-accuracy of 0.9995.

#training the model with the best hyperparameter on joined training and validation set:
model_knn = KNeighborsClassifier(n_neighbors = 1)
model_knn.fit(train_data, train_label)

y_pred_knn = model_knn.predict(test_data)
print("Accuracy KNN: {}".format(accuracy_score(test_label, y_pred_knn)))

##CNN

#Encoding the target variable for CNN
le=MultiLabelBinarizer()
le.fit(y_train.reshape(y_train.shape[0], 1))
y_train_enc = le.transform(y_train.reshape(y_train.shape[0], 1))
y_valid_enc = le.transform(y_valid.reshape(y_valid.shape[0], 1))
y_test_enc = le.transform(test_label.reshape(test_label.shape[0], 1))

#Reshaping the input data for the CNN
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid_cnn = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test_cnn = test_data.reshape(test_data.shape[0], 28, 28, 1)

#Creating the CNN with two convolutional layers for hyperparameter tuning
model = Sequential()
# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(24, activation='softmax'))

#Grid search hyperparameter tuning for CNN with two layers
with open('hyp_2l.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Batchsize', 'Epochs', 'Val_Accuracy', 'Time'])
    for batch_size in [128, 256, 512]:
        for epochs in [5, 10, 15]:
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            t = timer()
            history = model.fit(X_train_cnn, y_train_enc, batch_size=batch_size, epochs=epochs, validation_data=(X_valid_cnn, y_valid_enc))
            elapsed_time = timer() - t
            val_acc = history.history['val_accuracy'][-1]
            final = [batch_size, epochs, val_acc, elapsed_time]
            writer.writerow(final)
            
#Creating the CNN with one convolutional for hyperparameter tuning
model = Sequential()
# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(24, activation='softmax'))

#Grid search hyperparameter tuning for CNN with two layers
with open('hyp_1l.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Batchsize', 'Epochs', 'Val_Accuracy', 'Training-Time'])
    for batch_size in [128, 256, 512]:
        for epochs in [5, 10, 15]:
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            t = timer()
            history = model.fit(X_train_cnn, y_train_enc, batch_size=batch_size, epochs=epochs, validation_data=(X_valid_cnn, y_valid_enc))
            elapsed_time = timer() - t
            val_acc = history.history['val_accuracy'][-1]
            final = [batch_size, epochs, val_acc, elapsed_time]
            writer.writerow(final)

#All combinations had a great accuracy, 1 convolutional layer with Batchsize = 512 and 5 epochs had the best training time
            
#Preparing joined training and validation sets for training the final model
y_train_enc = le.transform(train_label.reshape(train_label.shape[0], 1))
X_train_cnn = train_data.reshape(train_data.shape[0], 28, 28, 1)

#Creating the final CNN with one convolutional layer
model_cnn = Sequential()
# convolutional layer
model_cnn.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
# convolutional layer
model_cnn.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model_cnn.add(MaxPool2D(pool_size=(2,2)))
model_cnn.add(Dropout(0.25))
# flatten output of conv
model_cnn.add(Flatten())
# hidden layer
model_cnn.add(Dense(500, activation='relu'))
model_cnn.add(Dropout(0.4))
model_cnn.add(Dense(250, activation='relu'))
model_cnn.add(Dropout(0.3))
# output layer
model_cnn.add(Dense(24, activation='softmax'))
model_cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model_cnn.fit(X_train_cnn, y_train_enc, batch_size=512, epochs=5)

#Predicting the test set
preds = model_cnn.predict(X_test_cnn)
preds = np.array([np.argmax(pred) for pred in preds])
y_pred_cnn = np.array([ i if i<9 else i+1 for i in preds])
print("Accuracy KNN: {}".format(accuracy_score(test_label, y_pred_cnn)))

### Task 2

data_task2 = np.load('data/test_images_task2.npy')

#function that locates the left edges of the images
def find_indexes(image):
    # The variances of differences of a column (higher for a column out of an image as its pixels are rondomly distributed)
    number_col = image.shape[1] #number of columns
    variances = []
    for i in range(number_col):
        diffs = np.diff(image[:,i]) #the changes in pixel values of two adjacent pixels within a column
        var_diffs = np.var(diffs) #variance of those changes
        variances.append(var_diffs) #all variances for each column  
    # We are looping an image (of 200 columns) through its columns and calculating the average variances of 
    # all the adjacent 28 columns, as one candidate image used in the model must have 28 columns.
    # Here, we calculate the average variances of each 28 * 28 images (in total 172 images) by taking the average of variances 
    # of each column they contain. The ones giving locally smallest variances are selected to be used for prediction
    avg_variances = np.array([])
    for i in range(number_col-28):
        img_variance = np.mean(variances[i:i+28])
        avg_variances = np.append(avg_variances, img_variance)
    # After seeing locally smallest variances are smaller than 1100 (by visualization), only those variances are taken to find 
    # the locally smallest ones among them otherwise we would have lots of locally smallest variances
    """
    import cv2
    import matplotlib.pyplot as plt
    plt.plot(avgs)
    plt.ylabel('variances of images by the first column')
    plt.show()
    """
    trimmed_avg_variances = avg_variances[avg_variances < 1200]
    # Finding the index of initial column of images giving locally smallest variances
    index_in_trimmed= argrelextrema(trimmed_avg_variances, np.less)
    min_variances = trimmed_avg_variances[index_in_trimmed]
    indexes = np.in1d(avg_variances, min_variances).nonzero()[0]
    # This part is to make sure that the difference between the indexes of first columns of two images cannot be smaller than 28.
    i = 0
    while i in range(len(indexes)-1):
        if indexes[i+1] - indexes[i] < 28:
            if min_variances[i] < min_variances[i+1]:
                indexes = np.delete(indexes, i+1)
            else:
                indexes = np.delete(indexes, i)
            i -= 1
        i += 1
    return indexes

#Function that uses image detection to separate the images contained in an input-image and then predicts each image using the CNN from task 1.
def im_detect_cnn(img):
    indices = find_indexes(img)
    if indices.size > 0:
        split_seq = np.array([img[:,j:j+28]for j in indices])
        preds = model_cnn.predict(split_seq)
        pred_c = np.argmax(preds, axis = 1)
        out = ""
        #Due to the encoding of the target, the labels obtained with argmax do not pay respect to the classes 9 and 25
        #Leading 0 for single-digit classes are added and classes above 8 are corrected
        for c in pred_c:
            if c < 9:
                out += str(0)+str(c)
            else:
                out += str(c+1)
    #To avoid mistakes with generating outputs, an arbitrary chosen 'fail'-code gets added if no image gets detected
    else:
        out = '05000811'
    return out

def im_detect_knn(img):
    indices = find_indexes(img)
    if indices.size > 0:
        split_seq = np.array([img[:,j:j+28]for j in indices])
        #The detected images are flattened to use them as input for the KNN-model
        if indices.size == 1:
            split_seq = np.array([split_seq.flatten()])
        else:
            split_seq = np.array([x.flatten() for x in split_seq])
        preds = model_knn.predict(split_seq)
        out = ""
        for c in preds:
            #for single-digit classes, a leading 0 is added.
            if c < 10:
                out += str(0)+str(c)
            else:
                out += str(c)
    #To avoid mistakes with generating outputs, an arbitrary chosen 'fail'-code gets added if no image gets detected
    else:
        out = '05000811'
    return out

#function that makes prediction for every 28*28 area of the input image using the CNN of task 1.
def proba_pred(img, threshold = 0.999999):
    p = model_cnn.predict(np.array([img[:,i:i+28] for i in range(172)]))
    c = [q.argmax()for q in p]
    prob =[q.max()for q in p]
    preds = []
    out = ''
    #Recursive function that checks for each area if the probability of the prediction is above a chosen threshold
    #If a prediction is approved, the next 27 pixels are skipped
    def find(c, prob, start = 0):
        for l,j,k in zip(range(start, len(c)), c[start:], prob[start:]):
            if l == 172:
                if k >= threshold:
                    preds.append(j)
                    return
                else:
                    return
            elif k >= threshold:
                preds.append(j)
                find(c, prob, start = l+28)
                return
    find(c, prob)
    if len(preds) > 0:
        #Due to the encoding of the target, the labels obtained with argmax do not pay respect to the classes 9 and 25
        #Leading 0 for single-digit classes are added and classes above 8 are corrected
        for res in preds:
            if res < 9:
                out += str(0) + str(res)
            else:
                out += str(res+1)
    #To avoid mistakes with generating outputs, an arbitrary chosen 'fail'-code gets added if no image gets detected
    else:
        out = '05000811'
    return out
            
#To make the final predictions, we used each model from task 1 with the detected images and the probability based prediction eith three different thesholds
with open('predictions.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    length = len(data_task2)
    for i in range(length):
        final = [im_detect_cnn(data_task2[i]),im_detect_knn(data_task2[i]), proba_pred(data_task2[i], threshold = 0.99), proba_pred(data_task2[i], threshold = 0.9999), proba_pred(data_task2[i], threshold = 1.0)]
        writer.writerow(final)
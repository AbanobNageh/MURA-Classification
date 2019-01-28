import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn as sk
import sklearn.metrics

cachePath = "inception+mobilenetCache/"  # Path of the folder that contains the .pkl cache files.
labelListPath = "MURA-v1.1/labelLists/"                         # Path of the folder that contains the label lists.
validationLabelListPath = "MURA-v1.1/validationLabelList/"      # Path of the folder that contains the validation label lists.

printConfusionMatrix = True
useAllDate = True
catagoryNumber = 1

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def getCategoryData(categoryNumber):
    """
    Read the transfer data of one of the seven categories from pkl files.
    :param listsDirectory:
        a Number from 1 to 7 that specifies the category to read from file,
        these numbers are:
        1 -> Elbow
        2 -> Finger
        3 -> Forarm
        4 -> Hand
        5 -> Humerus
        6 -> Shoulder
        7 -> Wrist
    :return:
        Returns the training data, the labels of the training data, the evaluation data and the labels of the evaluation data of the given catagory.
    """

    trainingData = None
    trainingLabelData = []
    testingData = None
    testingLabelData = []

    if categoryNumber == 1:
        trainingDataPath = cachePath +  "elbowList.pkl"
        trainingLabelDataPath = labelListPath + "elbowLabelList.txt"
        testingDataPath = cachePath + "validationElbowList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationElbowLabelList.txt"
    elif categoryNumber == 2:
        trainingDataPath = cachePath +  "fingerList.pkl"
        trainingLabelDataPath = labelListPath + "fingerLabelList.txt"
        testingDataPath = cachePath + "validationFingerList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationFingerLabelList.txt"
    elif categoryNumber == 3:
        trainingDataPath = cachePath +  "forarmList.pkl"
        trainingLabelDataPath = labelListPath + "forarmLabelList.txt"
        testingDataPath = cachePath + "validationForarmList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationForarmLabelList.txt"
    elif categoryNumber == 4:
        trainingDataPath = cachePath +  "handList.pkl"
        trainingLabelDataPath = labelListPath + "handLabelList.txt"
        testingDataPath = cachePath + "validationHandList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationHandLabelList.txt"
    elif categoryNumber == 5:
        trainingDataPath = cachePath +  "humerusList.pkl"
        trainingLabelDataPath = labelListPath + "humerusLabelList.txt"
        testingDataPath = cachePath + "validationHumerusList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationHumerusLabelList.txt"
    elif categoryNumber == 6:
        trainingDataPath = cachePath +  "shoulderList.pkl"
        trainingLabelDataPath = labelListPath + "shoulderLabelList.txt"
        testingDataPath = cachePath + "validationShoulderList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationShoulderLabelList.txt"
    elif categoryNumber == 7:
        trainingDataPath = cachePath +  "wristList.pkl"
        trainingLabelDataPath = labelListPath + "wristLabelList.txt"
        testingDataPath = cachePath + "validationWristList.pkl"
        testingLabelDataPath = validationLabelListPath + "validationWristLabelList.txt"
    
    with open(trainingDataPath, mode='rb') as file:
        obj = pickle.load(file)

    with open(testingDataPath, mode='rb') as file:
        obj2 = pickle.load(file)

    with open(trainingLabelDataPath) as file:
        for line in file:
            line = line.strip()
            if line == "1":
                trainingLabelData.append(1)
            else:
                trainingLabelData.append(0)

    with open(testingLabelDataPath) as file:
        for line in file:
            line = line.strip()
            if line == "1":
                testingLabelData.append(1)
            else:
                testingLabelData.append(0)

    trainingData = np.array(obj)
    trainingLabelData = np.array(trainingLabelData)
    testingData = np.array(obj2)
    testingLabelData = np.array(testingLabelData)

    return trainingData, trainingLabelData, testingData, testingLabelData

def getAllData():
    """
    Read all the transfer data of all of the seven categories from pkl files.
    :return:
        returns the training data, the labels of the training data, the evaluation data and the labels of the evaluation data of the given catagory.
    """

    trainingData = None
    trainingLabelData = None
    testingData = None
    testingLabelData = None

    for i in range(1, 8):
        training, trainingLabel, testing, testingLabel = getCategoryData(i)

        if i == 1:
            trainingData = training
            trainingLabelData = trainingLabel
            testingData = testing
            testingLabelData = testingLabel
        else:
            trainingData = np.concatenate((trainingData, training))
            trainingLabelData = np.concatenate((trainingLabelData, trainingLabel))
            testingData = np.concatenate((testingData, testing))
            testingLabelData = np.concatenate((testingLabelData, testingLabel))
    
    return trainingData, trainingLabelData, testingData, testingLabelData

# Read the transfer values of the MURA dataset.catagoryNumber
if useAllDate is False:
    trainingData, trainingLabelData, testingData, testingLabelData = getCategoryData(catagoryNumber)
else:
    trainingData, trainingLabelData, testingData, testingLabelData = getAllData()

# create a simple neural network to train the data.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model this is used when calculating 
if printConfusionMatrix is True:
    history = model.fit(trainingData, trainingLabelData, epochs=5, validation_split=0.2)
    predictions = model.predict_classes(testingData)
    tensor = tf.confusion_matrix(testingLabelData, predictions, num_classes = 2, dtype=tf.int32)
else:
    history = model.fit(trainingData, trainingLabelData, epochs=5, validation_data=[testingData, testingLabelData])

with tf.Session():
    if printConfusionMatrix is True:
        # print the confusion matrix.
        confusionMatrix = tf.Tensor.eval(tensor,feed_dict=None, session=None)
        classNames = ["Normal", "Abnormal"]
        plt.figure()
        plotConfusionMatrix(confusionMatrix, classNames)

if printConfusionMatrix is False:
    acc = np.array(history.history['acc'])
    loss = np.array(history.history['loss'])
    valAcc = np.array(history.history['val_acc'])
    valLoss = np.array(history.history['val_loss'])

    # print the average values.
    print("accuracy = " + str(np.average(acc)))
    print("loss = " + str(np.average(loss)))
    print("val accuracy = " + str(np.average(valAcc)))
    print("val loss = " + str(np.average(valLoss)))
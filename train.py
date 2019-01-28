import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import Inception as inc
import Mobilenet as mn
import Densenet as dn
import pickle
import sys

# load the pretrained models.
mobilenetModel = mn.Mobilenet()
inceptionModel = inc.Inception()
densenetModel = dn.Densenet()

def calculateTransferValues(listsDirectory):
    """
    Calculate the transfer values of each image in the MURA dataset using
    all three pretrained models and then concatenate these values together
    and finally writes them to cache files.
    :param listsDirectory:
        Directory for the lists that contain the paths of image files.
    :return:
        Nothing.
    """
    
    imagePaths = []
    
    # Get files that hold the image paths (ex. elbowList.txt).
    listNames = os.listdir(listsDirectory)
    for listName in listNames: 
        # Get the path of the file and set the cache path with the same name.
        listPath = listsDirectory + "/" + listName
        listName = listName[:-4]
        cachePath = "cache/" + listName + ".pkl"
        print(listPath)

        # Read the paths in the file and remove the \n character.
        with open(listPath) as file:
            imagePaths = file.readlines()
        imagePaths = [x.strip() for x in imagePaths]

        # Create an empty list to hold the transfer values.
        imageCount = len(imagePaths)
        transferValues = [None] * imageCount

        # Loop over all images.
        for i in range(imageCount):
            msg = "\r- Processing image: {0:>6} / {1}".format(i+1, imageCount)
            sys.stdout.write(msg)
            sys.stdout.flush()

            imagePath = imagePaths[i]

            # Get the transfer values of all three models.
            mobilenetTransferValue = mobilenetModel.getTransferValue(imagePath)
            inceptionTransferValue = inceptionModel.getTransferValue(imagePath)
            densenetTransferValue = densenetModel.getTransferValue(imagePath)

            # Concatenate the transfer values.
            result = np.concatenate((inceptionTransferValue, mobilenetTransferValue))
            transferValues[i] = np.concatenate((result, densenetTransferValue))
        
        print()

        # Clear the list after each group of images and cache the transfer values.
        imagePaths.clear()
        cacheTransferValues(transferValues, cachePath)
    
    # Close the models to clear any resorces they hold.
    mobilenetModel.close()
    inceptionModel.close()
    densenetModel.close()

def calculateInceptionTransferValues(listsDirectory):
    """
    Calculate the transfer values of each image in the MURA dataset using
    the inception model only and then writes them to cache files.
    :param listsDirectory:
        Directory for the lists that contain the paths of image files.
    :return:
        Nothing.
    """
    
    imagePaths = []

    # Get files that hold the image paths (ex. elbowList.txt).
    listNames = os.listdir(listsDirectory)
    for listName in listNames:
        # Get the path of the file and set the cache path with the same name.
        listPath = listsDirectory + "/" + listName
        listName = listName[:-4]
        cachePath = "inceptionCache/" + listName + ".pkl"
        print(listPath)

        # Read the paths in the file and remove the \n character.
        with open(listPath) as file:
            imagePaths = file.readlines()
        imagePaths = [x.strip() for x in imagePaths] 

        # Get the transfer values and cache them.
        getInceptionTransferValues(imagePaths, cachePath)
        imagePaths.clear()
    
    inceptionModel.close()

def calculateMobilenetTransferValues(listsDirectory):
    """
    Calculate the transfer values of each image in the MURA dataset using
    the mobilenet model only and then writes them to cache files.
    :param listsDirectory:
        Directory for the lists that contain the paths of image files.
    :return:
        Nothing.
    """

    imagePaths = []

    # Get files that hold the image paths (ex. elbowList.txt).
    listNames = os.listdir(listsDirectory)
    for listName in listNames:
        # Get the path of the file and set the cache path with the same name.
        listPath = listsDirectory + "/" + listName
        listName = listName[:-4]
        cachePath = "mobilenetCache/" + listName + ".pkl"
        print(listPath)

        # Read the paths in the file and remove the \n character.
        with open(listPath) as file:
            imagePaths = file.readlines()
        imagePaths = [x.strip() for x in imagePaths]

        # Get the transfer values and cache them. 
        getMobileNetTransferValues(imagePaths, cachePath)
        imagePaths.clear()
    
    mobilenetModel.close()

def calculateDensenetTransferValues(listsDirectory):
    """
    Calculate the transfer values of each image in the MURA dataset using
    the densenet model only and then writes them to cache files.
    :param listsDirectory:
        Directory for the lists that contain the paths of image files.
    :return:
        Nothing.
    """
    
    imagePaths = []

    # Get files that hold the image paths (ex. elbowList.txt).
    listNames = os.listdir(listsDirectory)
    for listName in listNames:
        # Get the path of the file and set the cache path with the same name.
        listPath = listsDirectory + "/" + listName
        listName = listName[:-4]
        cachePath = "densenetCache/" + listName + ".pkl"
        print(listPath)

        # Read the paths in the file and remove the \n character.
        with open(listPath) as file:
            imagePaths = file.readlines()
        imagePaths = [x.strip() for x in imagePaths]

        # Get the transfer values and cache them. 
        getDensenetTransferValues(imagePaths, cachePath)
        imagePaths.clear()
    
    densenetModel.close()

def getInceptionTransferValues(imagePaths, cachePath):
    """
    Calculate the transfer values of a group of images using
    the inception model only and then writes them to cache files.
    :param imagePaths:
        List of image paths.
    :param cachePath:
        The file to cache the transfer values in.
    :return:
        Nothing.
    """
    values = inceptionModel.getTransferValues(imagePaths)
    cacheTransferValues(values, cachePath)

def getMobileNetTransferValues(imagePaths, cachePath):
    """
    Calculate the transfer values of a group of images using
    the mobilenet model only and then writes them to cache files.
    :param imagePaths:
        List of image paths.
    :param cachePath:
        The file to cache the transfer values in.
    :return:
        Nothing.
    """
    
    values = mobilenetModel.getTransferValues(imagePaths)
    cacheTransferValues(values, cachePath)

def getDensenetTransferValues(imagePaths, cachePath):
    """
    Calculate the transfer values of a group of images using
    the densenet model only and then writes them to cache files.
    :param imagePaths:
        List of image paths.
    :param cachePath:
        The file to cache the transfer values in.
    :return:
        Nothing.
    """
    
    values = densenetModel.getTransferValues(imagePaths)
    cacheTransferValues(values, cachePath)

def cacheTransferValues(transferValues, cachePath):
    """
    Caches the transfer values in the given file using pickle.
    :param transferValues:
        The transfer values to be cached
    :param cachePath:
        The file to cache the transfer values in.
    :return:
        Nothing.
    """
    
    print("caching transfer values in: " + cachePath)
    if not os.path.exists(cachePath):
        with open(cachePath, mode='wb') as file:
            pickle.dump(transferValues, file)

calculateTransferValues("E:/MURA-v1.1/lists")  
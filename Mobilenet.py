import numpy as np
import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import sys

class Mobilenet:
    data_dir = "mobilenet-data/"                                  # Directory of the mobilenet data.
    path_graph_def = "mobilenet_v2_1.4_224_frozen.pb"             # The pb file name
    tensor_name_input_image = "input:0"                           # Name of the input tensor
    tensor_name_transfer_layer = "MobilenetV2/Logits/AvgPool:0"   # Name of the output tensor

    def __init__(self):
        """
        Initilizes the class by reading the model data from the pb file.
        :return:
            Nothing.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            path = os.path.join(self.data_dir, self.path_graph_def)
            with tf.gfile.GFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')
        
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)
        self.session = tf.Session(graph=self.graph)
    
    def close(self):
        """
        Closes the model and frees any resources in use.
        :return:
            Nothing.
        """

        self.session.close()

    def isEvenNumber(self, number):
        """
        Checks if the given number is even.
        :param number:
            Number to check.
        :return:
            True if the number is even, False otherwise.
        """

        return number % 2 == 0

    def getPadding(self, rowCount, columnCount, paddingSize = 672):
        """
        Caculate the padding nedded to resize the image to 244, 244 without distortion.
        :param rowCount:
            Number of rows in the image == hight.
        :param columnCount:
            Number of columns in the image == width.
        :param paddingSize:
            Vertical and horozontal size of the image after padding, must be larger than the hight and the width of the image,
            default is 672
        :return:
            The top, bottom, left and right padding that should be added to the image.
        """

        # Calculate the needed vertical and horizontal padding.
        verticalPadding = paddingSize - rowCount
        horizontalPadding = paddingSize - columnCount

        # Calculate the top, bottom, left and right padding.
        if self.isEvenNumber(verticalPadding):
            topPadding = int(verticalPadding/2)
            bottomPadding = int(verticalPadding/2)
        else:
            topPadding = int(verticalPadding/2)
            bottomPadding = int((verticalPadding/2)) - 1

        if self.isEvenNumber(horizontalPadding):
            leftPadding = int(horizontalPadding/2)
            rightPadding = int(horizontalPadding/2)
        else:
            leftPadding = int(horizontalPadding/2)
            rightPadding = int((horizontalPadding/2)) - 1

        return topPadding, bottomPadding, leftPadding, rightPadding

    def prepareImage(self, imagePath):
        """
        Processes the image before calculating the transfer values, it takes the following steps:
        reads the image, changes the color channels from BGR to RGB (because openCV reads it in BGR format), 
        adds padding to the image, resizes the image, change it to a numpy array and if needed changes the image
        shape to (1, ?, ?, 3)
        :param imagePath:
            Path of the image to prepare.
        :return:
            The image as a numpy array.
        """
        
        # Read the image and switch its color channel order from BGR to RGB.
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Add padding to the image to prevent it from being distorted when resized.
        rowCount, columnCount = image.shape[:2]
        topPadding, bottomPadding, leftPadding, rightPadding = self.getPadding(rowCount, columnCount)
        image = cv.copyMakeBorder(image, 
                                  top = topPadding, 
                                  bottom = bottomPadding, 
                                  left = leftPadding, 
                                  right = rightPadding, 
                                  borderType = cv.BORDER_REPLICATE)
        
        # Resize the image to 244, 244 as this is needed by the pretrained model.
        image = cv.resize(image, (224 , 224))
        image = np.array(image)

        # Change the shape of the image to (1, ?, ?, 3) as this is needed by the pretrained model.
        image = image[np.newaxis, :]
        return image

    def getTransferValue(self, imagePath):
        """
        Calculates the transfer values of a single image.
        :param imagePath:
            Path of the image to to calculate trransfer values for.
        :return:
            The transfer values as a numpy array.
        """
        
        image = self.prepareImage(imagePath)

        # Process the image and get the transfer values.
        feed_dict = {self.tensor_name_input_image: image}
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # Squeeze the transfer values to be 1D.
        transfer_values = np.squeeze(transfer_values)
        return transfer_values

    def getTransferValues(self, imagePaths):
        """
        Calculates the transfer values of a group of images.
        :param imagePaths:
            List of image paths to calculate transfer values for..
        :return:
            The transfer values as a numpy array.
        """
        
        # Create an empty list for the trasnfer values.
        imageCount = len(imagePaths)
        result = [None] * imageCount

        for i in range(imageCount):
            msg = "\r- Processing image: {0:>6} / {1}".format(i+1, imageCount)
            sys.stdout.write(msg)
            sys.stdout.flush()

            imagePath = imagePaths[i]
            image = self.prepareImage(imagePath)

            # Process the image and get the transfer values.
            feed_dict = {self.tensor_name_input_image: image}
            transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

            # Squeeze the transfer values to be 1D.
            transfer_values = np.squeeze(transfer_values)
            result[i] = transfer_values
        
        print()
        result = np.array(result)
        return result
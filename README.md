# MURA-Classification

A project that classifies the MURA dataset by using transfer learning, the pretrained models in this project are:
1. inception V3.
2. mobilenet.
3. densenet.

to test the project follow these steps:
1. download the code.
2. download the MURA V1.1 dataset and extract the folder 'MURA-v1.1' to the same directory as the project files.
3. download the pretrained models from [inception](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz "inception download link"), [mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet "mobilenet download page") and [densenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet "densenet download page").
4. use the instructions on [this page](https://github.com/huanzhang12/tensorflow-densenet-models "densenet instructions") to freeze the densenet model.
5. create 3 folders 'densenet-data', 'inception-data' and 'mobilenet-data' and add the files of the pretrained models tp these folders.
6. run the file 'preprocessMURA.py', this code splits the files 'train_image_paths.csv' and 'valid_image_paths.csv' into files, one file for each category of images in the dataset.
7. run the file 'train.py', this code will produce the transfer values of the MURA dataset using one or more of the pretrained models, the default is all three pretrained models. if more than one pretrained model is used, the result of the pretrained models of each image are concatenated, the final result will then be cached to pkl files, this might take a long time depending on your hardware.
8. run the file 'evaluateMURA.py', this file will use the produced pkl files in order to evaluate the training data using the validation data. 

a sample of the result by using the mobilenet and the inception models on the validation data.
![alt text](MURA-Classification/Confusion Matrix.png "Confusion Matrix")

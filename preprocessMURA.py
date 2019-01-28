# This file is used to seperate the .csv file that holds the MURA dataset paths into
# lists for each category.
import os

trainingFilePath = "MURA-v1.1/train_image_paths.csv"       # Path to the file "train_image_paths.csv"
trainingListSavePath = "MURA-v1.1/lists/"                  # Path to save the training images lists.
trainingLabelSavePath = "MURA-v1.1/labelLists/"            # Path to save the training images labels lists.

validationFilePath = "MURA-v1.1/valid_image_paths.csv"       # Path to the file "valid_image_paths.csv"
validationListSavePath = "MURA-v1.1/validationLists/"        # Path to save the validation images lists.
validationLabelSavePath = "MURA-v1.1/validationLabelList/"   # Path to save the validation images labels lists.

elbowList = []
elbowLabelList = []
fingerList = []
fingerLabelList = []
forarmList = []
forarmLabelList = []
handList = []
handLabelList = []
humerusList = []
humerusLabelList = []
shoulderList = []
shoulderLabelList = []
wristList = []
wristLabelList = []

def saveList(fileName, savePath, list):
    Directory = savePath + fileName

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    with open(Directory, 'w') as file:
        for line in list:
            file.write("%s\n" % line)

with open(trainingFilePath) as file:
    for line in file:
        line = line.strip()

        if "ELBOW" in line:
            elbowList.append(line)

            if "positive" in line:
                elbowLabelList.append("1")
            else:
                elbowLabelList.append("0")
        elif "FINGER" in line:
            fingerList.append(line)

            if "positive" in line:
                fingerLabelList.append("1")
            else:
                fingerLabelList.append("0")
        elif "FOREARM" in line:
            forarmList.append(line)

            if "positive" in line:
                forarmLabelList.append("1")
            else:
                forarmLabelList.append("0")
        elif "HAND" in line:
            handList.append(line)

            if "positive" in line:
                handLabelList.append("1")
            else:
                handLabelList.append("0")
        elif "HUMERUS" in line:
            humerusList.append(line)

            if "positive" in line:
                humerusLabelList.append("1")
            else:
                humerusLabelList.append("0")
        elif "SHOULDER" in line:
            shoulderList.append(line)

            if "positive" in line:
                shoulderLabelList.append("1")
            else:
                shoulderLabelList.append("0")
        elif "WRIST" in line:
            wristList.append(line)

            if "positive" in line:
                wristLabelList.append("1")
            else:
                wristLabelList.append("0")

saveList("elbowList.txt", trainingListSavePath, elbowList)
saveList("elbowLabelList.txt", trainingLabelSavePath, elbowLabelList)
saveList("fingerList.txt", trainingListSavePath, fingerList)
saveList("fingerLabelList.txt", trainingLabelSavePath, fingerLabelList)
saveList("forarmList.txt", trainingListSavePath, forarmList)
saveList("forarmLabelList.txt", trainingLabelSavePath, forarmLabelList)
saveList("handList.txt", trainingListSavePath, handList)
saveList("handLabelList.txt", trainingLabelSavePath, handLabelList)
saveList("humerusList.txt", trainingListSavePath, humerusList)
saveList("humerusLabelList.txt", trainingLabelSavePath, humerusLabelList)
saveList("shoulderList.txt", trainingListSavePath, shoulderList)
saveList("shoulderLabelList.txt", trainingLabelSavePath, shoulderLabelList)
saveList("wristList.txt", trainingListSavePath, wristList)
saveList("wristLabelList.txt", trainingLabelSavePath, wristLabelList)

elbowList.clear()
elbowLabelList.clear()
fingerList.clear()
fingerLabelList.clear()
forarmList.clear()
forarmLabelList.clear()
handList.clear()
handLabelList.clear()
humerusList.clear()
humerusLabelList.clear()
shoulderList.clear()
shoulderLabelList.clear()
wristList.clear()
wristLabelList.clear()

with open(validationFilePath) as file:
    for line in file:
        line = line.strip()

        if "ELBOW" in line:
            elbowList.append(line)

            if "positive" in line:
                elbowLabelList.append("1")
            else:
                elbowLabelList.append("0")
        elif "FINGER" in line:
            fingerList.append(line)

            if "positive" in line:
                fingerLabelList.append("1")
            else:
                fingerLabelList.append("0")
        elif "FOREARM" in line:
            forarmList.append(line)

            if "positive" in line:
                forarmLabelList.append("1")
            else:
                forarmLabelList.append("0")
        elif "HAND" in line:
            handList.append(line)

            if "positive" in line:
                handLabelList.append("1")
            else:
                handLabelList.append("0")
        elif "HUMERUS" in line:
            humerusList.append(line)

            if "positive" in line:
                humerusLabelList.append("1")
            else:
                humerusLabelList.append("0")
        elif "SHOULDER" in line:
            shoulderList.append(line)

            if "positive" in line:
                shoulderLabelList.append("1")
            else:
                shoulderLabelList.append("0")
        elif "WRIST" in line:
            wristList.append(line)

            if "positive" in line:
                wristLabelList.append("1")
            else:
                wristLabelList.append("0")

saveList("validationElbowList.txt", validationListSavePath, elbowList)
saveList("validationElbowLabelList.txt", validationLabelSavePath, elbowLabelList)
saveList("validationFingerList.txt", validationListSavePath, fingerList)
saveList("validationFingerLabelList.txt", validationLabelSavePath, fingerLabelList)
saveList("validationForarmList.txt", validationListSavePath, forarmList)
saveList("validationForarmLabelList.txt", validationLabelSavePath, forarmLabelList)
saveList("validationHandList.txt", validationListSavePath, handList)
saveList("validationHandLabelList.txt", validationLabelSavePath, handLabelList)
saveList("validationHumerusList.txt", validationListSavePath, humerusList)
saveList("validationHumerusLabelList.txt", validationLabelSavePath, humerusLabelList)
saveList("validationShoulderList.txt", validationListSavePath, shoulderList)
saveList("validationShoulderLabelList.txt", validationLabelSavePath, shoulderLabelList)
saveList("validationWristList.txt", validationListSavePath, wristList)
saveList("validationWristLabelList.txt", validationLabelSavePath, wristLabelList)

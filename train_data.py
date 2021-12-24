import cv2
import os
import numpy as np

#输入图像数据
imgA = cv2.imread('')
imgB = cv2.imread('')
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
imgAsize = imgA.shape
imgBsize = imgB.shape
if imgAsize != imgBsize:
    print(" size error!")
else:
    count = 0
    # 构建PM-Net网络训练样本
    for i in range(0, imgAsize[1] - 32, 32):
        for j in range(0, imgAsize[0] - 32, 32):
            train_tempA = imgA[j:(j+32), i:(i+32)]
            train_tempB = imgB[j:(j+32), i:(i+32)]
            cv2.imwrite("./img/Temp1/tempA/img" + str(count) + ".jpg", train_tempA)
            cv2.imwrite("./img/Temp1/tempB/img" + str(count) + ".jpg", train_tempB)
            count += 1
    train_sampleA = []
    for file in os.listdir("./img/traintemp/tempA"):
        train_sampleA.append("./img/traintemp/tempA" + "/" + file)
    train_sampleB = []
    for file in os.listdir("./img/traintemp/tempB"):
        train_sampleB.append("./img/traintemp/tempB" + "/" + file)
    train_sampleTrue = []
    train_sampleFalse = []
    train_sampleTrue = np.array([train_sampleA, train_sampleB])
    train_sampleFalse = np.array([train_sampleA, train_sampleB])
    np.random.shuffle(train_sampleFalse[0])
    np.random.shuffle(train_sampleFalse[1])
    for i in range(count):
        # PM-Net网络训练正样本
        train_stempATrue = cv2.imread(str(train_sampleTrue[0][i]), 0)
        train_stempBTrue = cv2.imread(str(train_sampleTrue[1][i]), 0)
        cv2.imwrite("./img/train/true/img" + str(i) + ".jpg",
                    cv2.merge([train_stempATrue, train_stempBTrue, np.zeros([32, 32], dtype="uint8")]))
        # PM-Net网络训练负样本
        tempAFalse = cv2.imread(str(train_sampleFalse[0][i]), 0)
        tempBFalse = cv2.imread(str(train_sampleFalse[1][i]), 0)
        cv2.imwrite("./img/train/false/img" + str(i) + ".jpg",
                    cv2.merge([tempAFalse, tempBFalse, np.zeros([32, 32], dtype="uint8")]))

    # PM-Net网络测试样本
    count = 0
    for i in range(16, imgAsize[1] - 32, 32):
        for j in range(16, imgAsize[0] - 32, 32):
            test_tempA = imgA[j:(j + 32), i:(i + 32)]
            test_tempB = imgB[j:(j + 32), i:(i + 32)]
            cv2.imwrite("./img/Temp2/tempA/img" + str(count) + ".jpg", test_tempA)
            cv2.imwrite("./img/Temp2/tempB/img" + str(count) + ".jpg", test_tempB)
            count += 1
    test_sampleA = []
    for file in os.listdir("./img/testtemp/tempA"):
        test_sampleA.append("./img/testtemp/tempA" + "/" + file)
    test_sampleB = []
    for file in os.listdir("./img/testtemp/tempB"):
        test_sampleB.append("./img/testtemp/tempB" + "/" + file)
    test_sampleTrue = []
    test_sampleFalse = []
    test_sampleTrue = np.array([test_sampleA, test_sampleB])
    test_sampleFalse = np.array([test_sampleA, test_sampleB])
    np.random.shuffle(test_sampleFalse[0])
    np.random.shuffle(test_sampleFalse[1])
    for i in range(count):
        test_tempATrue = cv2.imread(str(test_sampleTrue[0][i]), 0)
        test_tempBTrue = cv2.imread(str(test_sampleTrue[1][i]), 0)
        cv2.imwrite("./img/test/true/img" + str(i) + ".jpg",
                    cv2.merge([test_tempATrue, test_tempBTrue, np.zeros([32, 32], dtype="uint8")]))
        tempAFalse = cv2.imread(str(test_sampleFalse[0][i]), 0)
        tempBFalse = cv2.imread(str(test_sampleFalse[1][i]), 0)
        cv2.imwrite("./img/test/false/img" + str(i) + ".jpg",
                    cv2.merge([tempAFalse, tempBFalse, np.zeros([32, 32], dtype="uint8")]))

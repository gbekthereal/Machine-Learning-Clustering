# useful libraries
from sklearn import metrics
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
###############################################################################

#  .------------------.  #
#  .    FUNCTIONS     .  #
#  .------------------.  #

def calc_Purity(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity

def calc_Fmeasure(y_train, y_pred):
    fmeasure = metrics.f1_score(y_train, y_pred, average = "micro")
    return fmeasure

def parse_Data(dataDir):
    img_list = []
    for root, directories, filenames in os.walk(dataDir):
        for filename in filenames:
            if len(img_list) == 50:
                break
            else:
                if filename.endswith(".jpg"):
                    filei = os.path.join(root, filename)
                    img_list.append(filei)
    return img_list

def read_trainDataset():
    K0 = parse_Data("../input/facedata/train_data/2970")
    K1 = parse_Data("../input/facedata/train_data/10")
    K2 = parse_Data("../input/facedata/train_data/500")
    K3 = parse_Data("../input/facedata/train_data/1004")
    K4 = parse_Data("../input/facedata/train_data/2003")
    K5 = parse_Data("../input/facedata/train_data/3615")
    K6 = parse_Data("../input/facedata/train_data/106")
    K7 = parse_Data("../input/facedata/train_data/1131")
    K8 = parse_Data("../input/facedata/train_data/3190")
    K9 = parse_Data("../input/facedata/train_data/3250")
    return K0 + K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8 + K9 

def read_testDataset():
    testDataset = []
    testDataset = np.asarray(testDataset)
    for i in range(50):
        testDataset = np.append(testDataset, 0)

    for i in range(50):
        testDataset = np.append(testDataset, 1)

    for i in range(50):
        testDataset = np.append(testDataset, 2)

    for i in range(50):
        testDataset = np.append(testDataset, 3)

    for i in range(50):
        testDataset = np.append(testDataset, 4)

    for i in range(50):
        testDataset = np.append(testDataset, 5)

    for i in range(50):
        testDataset = np.append(testDataset, 6)

    for i in range(50):
        testDataset = np.append(testDataset, 7)

    for i in range(50):
        testDataset = np.append(testDataset, 8)

    for i in range(50):
        testDataset = np.append(testDataset, 9)
    return testDataset

def read_images(x):
    return np.asarray(Image.open(x)) / 255

def gray_scale(x):
    return np.dot(x[...,:3], [0.299, 0.587, 0.114])

def final_dataset(x):
    return x.reshape(x.shape[0] * x.shape[1])
###############################################################################

#  .------------------.  #
#  .      DATA        .  #
#  .------------------.  #

# creating the datasets
trainDataset = read_trainDataset()
testDataset = read_testDataset()

# reading the images from train dataset
for i in range(len(trainDataset)):
    trainDataset[i] = read_images(trainDataset[i])

# converting every rgb image to gray scale based on 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue formula 
for i in range(len(trainDataset)):
    trainDataset[i] = gray_scale(trainDataset[i])

# reshaping the data from (64, 64) to (4096, )
for i in range(len(trainDataset)):
    trainDataset[i] = final_dataset(trainDataset[i])

# giving the numpy form
trainDataset = np.asarray(trainDataset)
###############################################################################

#  .------------------.  #
#  . [METHOD 1]  PCA  .  #
#  .------------------.  #

M = [100, 50, 25]

pca0 = PCA(n_components = M[0])
trainDataset = pca0.fit_transform(trainDataset)

pca1 = PCA(n_components = M[1])
trainDataset1 = pca1.fit_transform(trainDataset)

pca2 = PCA(n_components = M[2])
trainDataset2 = pca2.fit_transform(trainDataset)
###############################################################################

#  .----------------------.  #
#  . [METHOD 2]  k-means  .  #
#  .----------------------.  #

kmeans = KMeans(n_clusters = 10)
y_pred_kmeans = kmeans.fit_predict(trainDataset)

Purity_kmeans = calc_Purity(testDataset, y_pred_kmeans)
F_Measure_kmeans = calc_Fmeasure(testDataset, y_pred_kmeans)
###############################################################################

#  .----------------------------------------------------.  #
#  . [METHOD 3]  agglomerative hierarchical clustering  .  #
#  .----------------------------------------------------.  #

hier_cluster = AgglomerativeClustering(n_clusters = 10, linkage = "ward")
y_pred_hier = hier_cluster.fit_predict(trainDataset)

Purity_hier = calc_Purity(testDataset, y_pred_hier)
F_Measure_hier = calc_Fmeasure(testDataset, y_pred_hier)
###############################################################################

#  .------------------.  #
#  .     PRINTS       .  #
#  .------------------.  #

print('--> k-means with euclidean distance <--')
print("Purity = {x}".format(x = Purity_kmeans), " and F-Measure = {y}".format(y = F_Measure_kmeans))

print("\n--> agglomerative hierarchical clustering <--")
print("Purity = {x}".format(x = Purity_hier), " and F-Measure = {y}".format(y = F_Measure_hier))
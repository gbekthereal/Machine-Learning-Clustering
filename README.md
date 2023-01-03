Implementing popular AI clustering algorithms like PCA, K-means and Agglomerative Hierarchical Clustering.

To run the code and you have to load the competition dataset 11785-Spring2021-HW2P2S1-Face-Classification for more information about the dataset, visit the site https://www.kaggle.com/c/11785-spring2021-hw2p2s1-face-classification.

The dataset got normalized as it is devided by 255 (read_images function) and the convertion of rgb image to gray color image uses the equation 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue (gray_scale function).
Also, tbe convertion of the image shape from (500, 64, 64) to (500, 4096) happens in final_dataset method.

In clustering the test labels for each data is no needed.

• Principal Component Analysis(PCA).
List M = [100,50,25] represents the values for the data dimension. So after the PCA method is used on the dataset, the data will form like (500, 100), (500, 50) 500, 25).

• K-means.
Implemented with euclidean distance and the number of clusters is 10.

• Agglomerative Hierarchical Clustering.
The linkage represents the distance and it will be used in trainDaatset as the Ward strategy.

• The evaluation of the performance of the methods can be computed by the Purity and F-measure. 

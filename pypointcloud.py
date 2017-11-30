import icp.py
import depthToCloud.py
import cv2
import numpy
import pcl

def fit(matrixA, matrixB):
	transformation = icp(matrixA,matrixB, (0,0,0), 13)
	return transformation*matrixA

def filter(matrix):
	fil = matrix.make_statistical_outlier_filter()
	fil.set_mean_k(50)
	fil.set_std_dev_mul_thresh(1.0)
	return fil.filter()

def imageToCloud(directory):
	img = cv2.imread(directory)
	return depthToCloud(img.shape)

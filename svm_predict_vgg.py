#coding:utf-8

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics

all_data_c1 = np.load('suozhiImagePool5Feature25.mat.npy')
all_data_c2 = np.load('zhenzhiImagePool5Feature25.mat.npy')

feature=[]
target=[]
for i in range(0,all_data_c1.shape[0]):
	feature.append(all_data_c1[i].reshape(all_data_c1.shape[1]))
	target.append(0)
for i in range(0,all_data_c2.shape[0]):
	feature.append(all_data_c2[i].reshape(all_data_c2.shape[1]))
	target.append(1)

feature_train, feature_local, target_train, target_local = train_test_split(feature, target, test_size=0.1, random_state=0)


clf = svm.SVC()
clf.fit(feature_train,target_train)
pred = clf.predict(feature_local)
accu = metrics.accuracy_score(target_local, pred)
print accu
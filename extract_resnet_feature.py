# coding=utf-8
import numpy as np 
import os
import scipy.io as sio
import mxnet as mx
import cv2
from collections import namedtuple

	
def get_image(url):
	# download and show the image
	img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
	if img is None:
		 return None
	# convert into format (batch, RGB, width, height)
	img = cv2.resize(img, (224, 224))
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 1, 2)
	img = img[np.newaxis, :]
	return img	

def extractFeature(imageBasePath, imageList,fe_mod, fea_mat):
	nd=2048
	features = np.ndarray((len(imageList),nd))
	num = 0
	for imagefile in imageList:
		imagefile_abs = os.path.join(imageBasePath, imagefile) # 图片绝对路径
		print imagefile_abs
		img = get_image(imagefile_abs)
		fe_mod.forward(Batch([mx.nd.array(img)]))
		features[num,:] = fe_mod.get_outputs()[0].asnumpy()
		num += 1
	np.save(fea_mat, features)
	
if __name__ == '__main__':
	path='http://data.mxnet.io/models/imagenet-11k/'
	[mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json'),
	 mx.test_utils.download(path+'resnet-152/resnet-152-0000.params')]
	sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
	mod = mx.mod.Module(symbol=sym, context=mx.cpu())
	mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
	mod.set_params(arg_params, aux_params)
	
	Batch = namedtuple('Batch', ['data'])
	all_layers = sym.get_internals()
	all_layers.list_outputs()[-10:]
	
	fe_sym = all_layers['flatten0_output']
	fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
	fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
	fe_mod.set_params(arg_params, aux_params)

	# 图片存放路径
	trainImageBasePath = '/root/data/images/suozhi/'
	testImageBasePath = '/root/data/images/zhenzhi/'
	#遍历目录获取文件列表
	trainImageList=[]
	testImageList=[]
	subfolders = [ fold for fold in os.listdir(trainImageBasePath)]
	for subfolder in subfolders:
		workspace_folder = os.path.join(trainImageBasePath,subfolder)
		for filename in os.listdir(workspace_folder):
			trainImageList.append(subfolder+"/"+filename)
	subfolders = [ fold for fold in os.listdir(testImageBasePath)]
	for subfolder in subfolders:
		workspace_folder = os.path.join(testImageBasePath,subfolder)
		for filename in os.listdir(workspace_folder):
			testImageList.append(subfolder+"/"+filename)
	extractFeature(trainImageBasePath,trainImageList,fe_mod,"resnet-152_suozhi.npy")
	extractFeature(testImageBasePath,testImageList,fe_mod,"resnet-152_zhenzhi.npy")
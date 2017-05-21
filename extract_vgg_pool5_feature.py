# coding=utf-8
import numpy as np 
import sys
import os
import struct
import scipy.io as sio    

caffe_root = '/root/Data/slu/caffe/caffe-6-11/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# deployPrototxt = ''
# modelFile = ''
# meanFile = '' # 也可以自己生产

# 需要提取的图像列表
# imageListFile = ''
# imageBasePath = ''

gpuIndex = 2

# 初始化函数的相关操作
def initialize(deployPrototxt, modelFile):
	print 'initializing ...'
	caffe.set_mode_gpu()
	caffe.set_device(gpuIndex)
	net = caffe.Net(deployPrototxt, modelFile, caffe.TEST)
	return net

# 提取特征并保存为相应的文件
# 参数说明：
#	imageBasePath:图片存放路径
#	imageList: 文件名称列表
#	net: 网络
#	fea_mat: 要保存的mat文件名称
#	nd:	pool5层特征维度
def extractFeature(imageBasePath, imageList, net, fea_mat, nd):
	# 对输入数据做出相应的调整
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #设定图片的shape格式(1,3,224,224)
	transformer.set_transpose('data', (2, 0, 1)) #改变维度的顺序，由原始图片(224,224,3)变为(3,224,224)  
	# transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # 减去均值，如果前面训练模型时如果没有减去均值，这儿就不用
	transformer.set_raw_scale('data', 255) # 缩放到[0,255]之间
	transformer.set_channel_swap('data', (2, 1, 0)) # 交换通道，将图片由RGB变为GBR

	# 设置batchsize,如果图片较多就设置合适的batchsize
	batchsize = 1
	net.blobs['data'].reshape(batchsize, 3, 224, 224)

	mean_value = np.array([104, 117, 123], dtype=np.float32) # 设置均值(由他人提供的数值...)
	mean_value = mean_value.reshape([3, 1, 1])

	num = 0
	fea = np.ndarray(shape=(len(imageList),nd,1,1))

	for imagefile in imageList:
		imagefile_abs = os.path.join(imageBasePath, imagefile) # 图片绝对路径
		print imagefile_abs
		im = caffe.io.load_image(imagefile_abs) # 加载图片
		net.blobs['data'].data[...] = transformer.preprocess('data', im) - mean_value# 执行上面的图片预处理操作，减去均值。将图片载入到blob中

		out = net.forward() # 执行测试

		tmp = net.blobs['pool5'].data # 抽取pool5的特征
		fea[num,:,:,:] = tmp
		num += 1
		#by zhu on 2017.5.11
		np.save(fea_mat+".npy", fea)
        #sio.savemat(fea_mat, {'pool5fea':fea}) # 保存成.mat文件


# 读取文件列表
def readImageList(imageListFile):
	imageList = []
	with open(imageListFile, 'r') as fi:
		while(True):
			line = fi.readline().strip('\n')
			if not line:
				break
			imageList.append(line)
	print 'read imageList done image num ', len(imageList)
	return imageList

if __name__ == '__main__':
	# 图片存放路径
	# class 1 image
        trainImageBasePath = '/root/data/images/suozhi/'
	# class 2 image
        testImageBasePath = '/root/data/images/zhenzhi/'

	# 网络结构
	deployPrototxt25 = '/root/Data/slu/small_grain/vgg_svm/ThiNet/0.25/deploy.prototxt'
	deployPrototxt5 = '/root/Data/slu/small_grain/vgg_svm/ThiNet/0.5/deploy.prototxt'

	# 网络模型
	modelFile25 = '/root/Data/slu/small_grain/vgg_svm/ThiNet/0.25/_iter_120000.caffemodel'
	modelFile5 = '/root/Data/slu/small_grain/vgg_svm/ThiNet/0.5/ThiNet-GAP.caffemodel'

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
	
	# 抽取特征
	# vgg 2.5
	net = initialize(deployPrototxt25, modelFile25)
	extractFeature(trainImageBasePath, trainImageList, net, './suozhiImagePool5Feature25.mat', 256)
	extractFeature(testImageBasePath, testImageList, net, './zhenzhiImagePool5Feature25.mat', 256)

	# vgg 5
	net = initialize(deployPrototxt5, modelFile5)
	extractFeature(trainImageBasePath, trainImageList, net, './suozhiImagePool5Feature5.mat', 512)
	extractFeature(testImageBasePath, testImageList, net, './zhenzhiImagePool5Feature5.mat', 512)

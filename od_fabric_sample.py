#coding:utf-8

import time
import numpy as np
import falconn
import os
import sys
from sys import argv
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# 图片存放路径
# class 1 image
trainImageBasePath = '/media/lee/data/zhu/images_old/suozhi/'
# class 2 image
testImageBasePath = '/media/lee/data/zhu/images_old/zhenzhi/'

# 网络结构
deployPrototxt5 = './ThiNet/0.5/deploy.prototxt'

# 网络模型
modelFile5 = './ThiNet/0.5/ThiNet-GAP.caffemodel'


gpuIndex = 0

# 初始化函数的相关操作
def initialize(deployPrototxt, modelFile):
    print 'initializing ...'
    caffe.set_mode_gpu()
    caffe.set_device(gpuIndex)
    net = caffe.Net(deployPrototxt, modelFile, caffe.TEST)
    return net
    
# 提取特征并保存为相应的文件
# 参数说明：
#    imageBasePath:图片存放路径
#    imageList: 文件名称列表
#    net: 网络
#    fea_mat: 要保存的mat文件名称
#    nd:    pool5层特征维度
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
    label = np.ndarray(shape=(len(imageList),2),dtype=object)

    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile) # 图片绝对路径
        im = caffe.io.load_image(imagefile_abs) # 加载图片
        net.blobs['data'].data[...] = transformer.preprocess('data', im) - mean_value# 执行上面的图片预处理操作，减去均值。将图片载入到blob中

        out = net.forward() # 执行测试

        tmp = net.blobs['pool5'].data # 抽取pool5的特征
        tmp_label=imagefile_abs.split("/")[-2]
        fea[num,:,:,:] = tmp
        label[num] = np.array([tmp_label,imagefile_abs])
        num += 1
        #by zhu on 2017.5.11
    np.save(fea_mat+".npy", fea)
    np.save(fea_mat+"_label.npy", label)
            
 
def exportTrainData():
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
    # vgg 5
    net = initialize(deployPrototxt5, modelFile5)
    extractFeature(trainImageBasePath, trainImageList, net, './suozhiImage', 512)
    extractFeature(testImageBasePath, testImageList, net, './zhenzhiImage', 512)

def exportTestData(imagefile_abs,nd):
    net = initialize(deployPrototxt5, modelFile5)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1)) 
    transformer.set_raw_scale('data', 255) 
    transformer.set_channel_swap('data', (2, 1, 0)) 
    batchsize = 1
    net.blobs['data'].reshape(batchsize, 3, 224, 224)
    mean_value = np.array([104, 117, 123], dtype=np.float32) 
    mean_value = mean_value.reshape([3, 1, 1])
    fea = np.ndarray(shape=(1,nd,1,1))
    im = caffe.io.load_image(imagefile_abs) 
    net.blobs['data'].data[...] = transformer.preprocess('data', im) - mean_value
    out = net.forward() 
    tmp = net.blobs['pool5'].data 
    fea[0] = tmp
    np.save("testImage.npy", fea)
    
def findTestPic(npy_path,t,K):
    result_list=[]
    all_data_test = np.load(npy_path)
    for i in range(0,all_data_test.shape[0]):
        u=all_data_test[i].reshape(all_data_test.shape[1])
        res = t.find_k_nearest_neighbors(u, K)
        result_list.append(res)
    return result_list
        
    
if __name__ == '__main__':
    if os.path.exists("./suozhiImage.npy") and os.path.exists("./zhenzhiImage.npy") and os.path.exists("./suozhiImage_label.npy") and os.path.exists("./zhenzhiImage_label.npy"):
        pass
    else:
        exportTrainData()
            
      
    all_data_c1 = np.load('suozhiImage.npy')
    all_data_c2 = np.load('zhenzhiImage.npy')
    all_target_c1 = np.load('suozhiImage_label.npy')
    all_target_c2 =  np.load('zhenzhiImage_label.npy')
    feature=[]
    target=[]
    for i in range(0,all_data_c1.shape[0]):
        feature.append(all_data_c1[i].reshape(all_data_c1.shape[1]))
        target.append(all_target_c1[i][0])
    for i in range(0,all_data_c2.shape[0]):
        feature.append(all_data_c2[i].reshape(all_data_c2.shape[1]))
        target.append(all_target_c2[i][0])
    n = len(feature)
    d = len(feature[0])
    p = falconn.get_default_parameters(n, d)
    t = falconn.LSHIndex(p)
    dataset = np.array(feature)
    t.setup(dataset)
    K=1
    
#"/media/lee/data/zhu/images_old/suozhi/208/2017-04-06-161609.jpg"
    exportTestData(argv[1],512)
    all_res = findTestPic("./testImage.npy",t,K)
    print "Image Label:"
    for r in all_res[0]:
        print target[r]
# -*- coding: utf-8 -*-
import caffe_pb2

import numpy as np

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')

# deploy文件
MODEL_FILE = 'mnist/deploy.prototxt'
# 预先训练好的caffe模型
PRETRAIN_FILE = 'mnist/lenet_iter_9380.caffemodel'
WRITE_FILE ='mnist/my_lenet.caffemodel'

# 保存参数的文件
params_txt = 'params.txt'
pf = open(params_txt, 'w')
input=open(PRETRAIN_FILE,'rb')

param=caffe_pb2.NetParameter()
param.ParseFromString(input.read())

#def read_caffemodel(caffefile):
print param
pf.close
input.close()

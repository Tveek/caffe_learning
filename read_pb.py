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
WRITE_FILE2='mnist/my_lenet2.caffemodel'

# 保存参数的文件
params_txt = 'params.txt'
#pf = open(params_txt, 'w')
input=open(WRITE_FILE,'rb')
output=open(WRITE_FILE2,'wb')

param=caffe_pb2.NetParameter()
param.ParseFromString(input.read())

param_name="InnerProduct2"
#def read_caffemodel(caffefile):

net=caffe_pb2.NetParameter()
net=param
#netlayer=net.layer.add()

for layers in param.layer:
    if layers.name==param_name:
        print layers.blobs[0].data[0]
        layers.blobs[0].data[0]=0
        #netlayer = layers

net_str=net.SerializeToString()

output.write(net_str)

#成功读取并修改，现在生成

#pf.close
input.close()
output.close()

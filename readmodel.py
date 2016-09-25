# -*- coding: utf-8 -*-
import caffe

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

# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
net.save(WRITE_FILE)
# 遍历每一层
#for param_name in net.params.keys():
param_name="InnerProduct2"
# 权重参数
weight = net.params[param_name][0].data
# 偏置参数
bias = net.params[param_name][1].data

# 该层在prototxt文件中对应“top”的名称
pf.write(param_name)
pf.write('\n')

# 写权重参数
pf.write('\n' + param_name + '_weight:\n\n')
# 权重参数是多维数组，为了方便输出，转为单列数组
#weight.shape = (-1, 1)

for i in range(0,9):
    for w in weight[i]:
        pf.write('%ff, ' % w)

    # 写偏置参数
    pf.write('\n\n' + str(i) + '_bias:\n\n')
    # 偏置参数是多维数组，为了方便输出，转为单列数组
    bias.shape = (-1, 1)
#for b in bias:
    pf.write('%ff\n ' % bias[i])

pf.write('\n\n')

pf.close
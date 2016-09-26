# -*- coding: utf-8 -*-
import caffe_pb2

class Result:
    param = caffe_pb2.NetParameter()
    def __init__(self,infile):
        input = open(infile, 'rb')
        Result.param.ParseFromString(input.read())

    def getlayer(self,param_name):
        for layers in Result.param.layer:
            if layers.name == param_name:
                return layers

    def getdata(self,param_name,n):
        for layers in Result.param.layer:
            if layers.name == param_name:
                return layers.blobs


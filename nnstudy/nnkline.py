#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-28 16:28:50
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os,sys

import json
import timetool


#当前对象主要为对k线数据分类的神经网络对象,包括数据分类网络训练，和网络保存使用

class KlineNNTool(object):
    """docstring for KlineNNTool"""
    def __init__(self, datpth,nnconfigpth):
        self.datapth = datpth               #训练用的数据文件所在目录
        self.nnconfigpth = nnconfigpth


        self.nnSaveFilePth = None

        self.m5NN = None                    #5分钟神经网络,
        self.h1NN = None                    #1小时神经网络
        self.h4NN = None                    #4小时神经网络
        self.h12NN = None                   #12小时网络
        self.h24NN = None                   #24小时网络
        
        self.fenleix = 10                   #分类x行数量    
        self.fenleiy = 10                   #分类y列数量
        self.fenleiIndex = 0                #分类编号

        #最后一次训练的k线数据时间,初始化时，会查找当前的新数据是否比原来数据新，如果新，则使用新数据对网络进行更新训练
        self.lastKlineTime = 0              

        self.initObj()

    #初始化神经网络，如果没有网络则训练一个网络
    def initObj(self):
        pass

    def create5MinNN(self):      #创建5分钟自编码器分类网络,如果文件中已存在网络，则先加载再进行训练
        pass

    def create1hourNN(self):     #创建1小时自编码器分类网络，如果文件中已存在网络，则先加载再进行训练
        pass

    def create4hourNN(self):     #创建4小时自编码器分类网络，如果文件中已存在网络，则先加载再进行训练
        pass

    def create12hourNN(self):    #创建12小时自编码器分类网络，如果文件中已存在网络，则先加载再进行训练
        pass

    def create24hourNN(self):    #创建24小时自编码器分类网络，如果文件中已存在网络，则先加载再进行训练
        pass

    def training5minNN(self):    #训练5分钟自编码分类网络，并保存到文件
        pass

    def training1hourNN(self):   #训练1小时自编码分类网络，并保存到文件
        pass

    def training4hourNN(self):   #训练4小时自编码分类网络，并保存到文件
        pass

    def training12hourNN(self):  #训练12小时自编码分类网络，并保存到文件
        pass

    def training24hourNN(self):  #训练24小时自编码分类网络，并保存到文件
        pass

    def trainingAllNN(self):     #训练总体分类神经网络，并保存到文件,输入为5分钟分类，1小时分类，4小时分类，12小时分类，24小时分类
        pass

    def getAddNewDataXY(self,data):   #添加一次数据到网络，返回神经网络分类坐标
        pass

    def getAddNewDataIndex(self,data):   #添加一次数据到网络，返回当前分类状态值
        pass

def main():
    pass

def test():
    pass

    # print sntimes
#测试
if __name__ == '__main__':
    # args = sys.argv
    # fpth = ''
    # if len(args) == 2 :
    #     if os.path.exists(args[1]):
    #         fpth = args[1]
    #     else:
    #         print "请加上要转码的文件路径"
    # else:
    #     print "请加上要转码的文件路径"
    #     main()
    main()
    

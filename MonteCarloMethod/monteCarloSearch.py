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

class MonteCarloSearch(object):
    """docstring for KlineNNTool"""
    def __init__(self, mangeOjb,treepth):

        self.mange = mangeOjb                #强化学习对象，强化学习对象中包含两个对象，一个是当前蒙特卡罗搜索树对象，一个是神经网络数据分类对象

        self.treepth = treepth               #蒙特卡罗树数据保存路径

        self.QStree = None                   #蒙特卡罗搜索树，使用字典来进行参数保存，当保存在文件时使用json格式
        self.States = []                     #从开始到现在的动作状态序列表
        self.actions = []                    #从开始到现在的动作序列表
        self.stateRs = []                    #状态之间转变时的单次收益序列表
        self.lastKline = None

        self.mcDeep = 0                      #蒙特卡罗树最大深度

        self.treeSearchCount = 0             #树搜索次数

        self.lastTreePth = []                #卡罗树最后一次数据搜索路径

        self.lastSearchTime = 0              #最后一次搜索时的数据输入时间,当下一次数据时间与这个时间差一个数据时间时，会对树进行一次更新

        self.initObj()

        #所有操作，开多，开空，不操作，卖出平多，买入平空
        #每次操作要根据之前的状态来判断是否可以下当前单，下单优先顺序为，
        #不操作，（开多，开空），(平多，平空),
        #当有一个多(空)存在时，要作相反操作时，先考虑开一个相反的单，当相反的单已存在，则先考虑平掉当前方向的单
        self.ops = ['openBuy','openSell','wait','sellForCloseBuy','buyForCloseSell']     

        self.onceOp = 100                   #每次操作下单数值，


    #初始化蒙特卡罗树
    def initObj(self):
        if os.path.exists(self.treepth):
            f = open(self.treepth,'r')
            jsonstr = f.read()
            f.close()
            self.QStree = json.loads(jsonstr)
        else:
            self.QStree = {}

    #更新当前树
    def updateTree():
        pass

    #搜索下一次应该进行的操作,同时更新搜索树,index:当前编号,kline:当前k线数字
    #所作的操作,会把当前数据结果输入到QStree和actions以方便进行下一次查找
    def searchWithIndex(self,satateIndex,kline):    
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
    

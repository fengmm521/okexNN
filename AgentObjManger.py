#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-28 16:28:50
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os,sys

import json
import timetool


#强化学习智能体,包含蒙特卡罗搜索树对象，kline数据预处理工具对象，神经网络数据分批分类对象
class AgentObj(object):
    """docstring for KlineNNTool"""
    def __init__(self, nnconfigflie,treepth,nnpth):

        self.nnObj = None                #神经网络分类对象

        self.treeOjb = None              #蒙特卡罗树搜索对象

        self.dataTool = None             #k线数据处理工具

        self.treeSavePth = treepth       #树参数保存路径

        self.nnSavePth = nnpth           #神经网络分类对象保存路径

        self.initObj()

    #初始化智能体,创建蒙特卡罗搜索树对象，k线处理工具对象，神经网络数据分类对象
    #1.先初始化k线数据，得到基本训练用的k线数据
    #2.把基本k线数据作处理，使得数据可以用来训练网络，再使用数据对神经网络分类器进行初始化训练，得到可以工作的分类网络工具
    #3.把初始数据送入第二步训练好的分类网络，得到数据的分类状态值
    #4.把有状态的初始数据送入蒙特卡罗对象，并使用随机的动作作多次训练，以得到可以预测成功率的蒙特卡罗树
    #到这里智能体初始化完成，已经可以工作了
    def initObj(self):
        pass

    #智能体进行一次数据更新
    #1.从数据工具中获取一个新数据，送入神经网络进行分类
    #2.将分类后的数据送入蒙特卡罗树，得到最优操作动作
    #3.对数据作出蒙特卡罗树的最优动作，并进行下一次数据更新
    #当数据工具中已经是最后一次数据时，要取新数据，需要一个等待新数据生成后获取间隔，所以下边数据更新方法会返回一个下次更新数据的时间
    def getNextData(self,dat):
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
    

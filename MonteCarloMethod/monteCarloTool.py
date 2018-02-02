#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-28 16:28:50
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os,sys

import json
from magetool import timetool


alldatapth = '../data/nnout/allout.json'

if not os.path.exists('mtcldata'):
    os.mkdir('mtcldata')

def getallData():
    f = open(alldatapth,'r')
    jstr = f.read()
    f.close()
    datas = json.loads(jstr)
    return datas

def saveDicWithJson(dic,jsonpth):
    ostr = json.dumps(dic)
    f = open(jsonpth,'w')
    f.write(ostr)
    f.close()

def saveDicWithLineTxt(dic,txtpth):
    ostr = ''

    for k in dic.keys():
        ostr += str(dic[k]) + '\n'

    ostr = ostr[:-1]
    f = open(txtpth,'w')
    f.write(ostr)
    f.close()

# [203703, [35.24329891204834, -206.66519498825073], [1510679400000, 66.74, 66.761, 66.507, 66.663, 48378.0, 7256.860988190463], '2017-11-15 01:10:00']
#时间戳，开，高，低，收，交易量
def findEque(datas):
    eques = {}
    for d in datas:
        tmpd = [d[0],[int(d[1][0]/50),int(d[1][1]/50)],d[2][4],d[3]]
        if not (d[0] in eques.keys()):
            eques[d[0]] = [tmpd]
        else:
            eques[d[0]].append(tmpd)
    return eques 

def main():
    datas = getallData()
    equs = findEque(datas)
    print len(equs)
    jsonpth = 'mtcldata/equedic.json'
    saveDicWithJson(equs, jsonpth)
    txtpth = 'mtcldata/equedic.txt'
    saveDicWithLineTxt(equs, txtpth)

def test():
    pass

    # print sntimes
#测试
if __name__ == '__main__':

    main()
    

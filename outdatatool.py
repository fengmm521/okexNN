#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-28 16:28:50
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os,sys

import json
import math
from magetool import timetool

BaseKlinePth = 'data/kline.txt'
koutpth = 'data/nnout/kline5out.json'

def read5MimKline():
    f = open(BaseKlinePth,'r')
    lines = f.readlines()
    f.close()
    kdats = []
    for l in lines:
        tmpl = l.replace('\n','').replace('\r','')
        tmpdat = json.loads(tmpl)
        kdats.append(tmpdat)

    return kdats

def readout5Data():
    f = open(koutpth,'r')
    tmpstr = f.read()
    f.close()
    datas = json.loads(tmpstr)
    return datas

def getLeath(x,y):
    l = math.sqrt(x*x + y*y)
    return l

def main():
    koutdatas = readout5Data()
    dlen = len(koutdatas)
    klinedats = read5MimKline()[-dlen:]
    
    print len(klinedats),len(koutdatas)

    maxy = [None,None]
    miny = [None,None]
    maxx = [None,None]
    minx = [None,None]

    max_xy = [None,None]
    min_xy = [None,None]
    max_xy_ = [None,None]
    min_xy_ = [None,None]

    ym = 0
    ys = 1000
    xm = 0
    xs = 1000
    lm = 0
    ls = 10000
    for n in range(len(koutdatas)):
        d = koutdatas[n]
        k = klinedats[n]
        if d[0] > xm:
            xm = d[0]
            maxx[0] = d
            maxx[1] = k
        if d[0] < xs:
            xs = d[0]
            minx[0] = d
            minx[1] = k
        if d[1] > ym:
            ym = d[1]
            maxy[0] = d
            maxy[1] = k
        if d[1] < ys:
            ys = d[1]
            miny[0] = d
            miny[1] = k
        lth = getLeath(d[0], d[1])
        if lth > lm:
            lm = lth
            max_xy[0] = d
            max_xy[1] = k
        if lth < ls:
            min_xy[0] = d
            min_xy[1] = k

    print 'minx:',minx,timetool.getNowDate(int(minx[1][0])/1000)
    print 'maxx:',maxx,timetool.getNowDate(int(maxx[1][0])/1000)
    print 'miny:',miny,timetool.getNowDate(int(miny[1][0])/1000)
    print 'maxy:',maxy,timetool.getNowDate(int(maxy[1][0])/1000)
    print 'min_xy:',min_xy,timetool.getNowDate(int(min_xy[1][0])/1000)
    print 'max_xy:',max_xy,timetool.getNowDate(int(max_xy[1][0])/1000)
    
    


    # print sntimes
#测试
if __name__ == '__main__':
    main()

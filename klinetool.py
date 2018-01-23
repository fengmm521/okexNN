#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-28 16:28:50
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os,sys

import json
import timetool

#获取脚本路径
def cur_file_dir():
    pathx = sys.argv[0]
    tmppath,_file = os.path.split(pathx)
    if cmp(tmppath,'') == 0:
        tmppath = sys.path[0]
    #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(tmppath):
        return tmppath
    elif os.path.isfile(tmppath):
        return os.path.dirname(tmppath)
    
#获取父目录
def GetParentPath(strPath):
    if not strPath:
        return None;
    lsPath = os.path.split(strPath);
    if lsPath[1]:
        return lsPath[0];
    lsPath = os.path.split(lsPath[0]);
    return lsPath[0];

#获取目录下的所有类型文件
def getAllExtFile(pth,fromatx = ".erl"):
    jsondir = pth
    jsonfilelist = []
    for root, _dirs, files in os.walk(jsondir):
        for filex in files:          
            #print filex
            name,text = os.path.splitext(filex)
            if cmp(text,fromatx) == 0:
                jsonArr = []
                rootdir = pth
                dirx = root[len(rootdir):]
                pathName = dirx +os.sep + filex
                jsonArr.append(pathName)
                (newPath,_name) = os.path.split(pathName)
                jsonArr.append(newPath)
                jsonArr.append(name)
                jsonfilelist.append(jsonArr)
            elif fromatx == ".*" :
                jsonArr = []
                rootdir = pth
                dirx = root[len(rootdir):]
                pathName = dirx +os.sep + filex
                jsonArr.append(pathName)
                (newPath,_name) = os.path.split(pathName)
                jsonArr.append(newPath)
                jsonArr.append(name)
                jsonfilelist.append(jsonArr)
    return jsonfilelist


#获取一个目录下的所有子目录路径
def getAllDirs(spth):
    files = getAllExtFile(spth,'.*')
    makedirstmp = []
    isOK = True
    # 分析所有要创建的目录
    for d in files:
        if d[1] != '/' and (not d[1] in makedirstmp): #创建未创建的目录层级
            tmpdir = d[1][1:]
            tmpleves = tmpdir.split('/')
            alldirs = getAllLevelDirs(tmpleves)
            for dtmp in alldirs:
                if not dtmp in makedirstmp:
                    makedirstmp.append(dtmp)
    return makedirstmp
#获取目录下的所有文件路径
def getAllFiles(spth,fromatx = '.*'):
    files = getAllExtFile(spth,fromatx)
    makedirstmp = []
    isOK = True
    # 分析所有要创建的目录
    for d in files:
        makedirstmp.append(d[0])
    return makedirstmp


def isFile(filename):
    try:
        with open(filename) as f:
            return True
    except IOError:
        return False

def finddir(arg,dirname,filenames):
    name,text = os.path.split(dirname)
    dirnametmp = str(dirname)
    if text and text[0] == '.':
        print dirname
        print filenames
        os.system('rm -r %s'%(dirname))
        return
    elif filenames:
        for f in filenames:
            if f[0] == '.' and isFile(dirname + f):
                fpthtmp = dirname + f
                if f.find(' '):
                    nf = f.replace(' ','\ ')
                    fpthtmp = dirname + nf
                print dirname + f
                os.system('rm  %s'%(fpthtmp))

#删除所有pth目录下的所有"."开头的文件名和目录名
def removeProjectAllHideDir(pth):
    alldirs = getAllDirs(pth)
    if not ('/' in alldirs):
        alldirs.append('/')
    for d in alldirs:
        tmpth = pth + d
        os.path.walk(tmpth, finddir, 0)



#获取一个路径中所包含的所有目录及子目录
def getAllLevelDirs(dirpths):
    dirleves = []
    dirtmp = ''
    for d in dirpths:
        dirtmp += '/' + d
        dirleves.append(dirtmp)
    return dirleves

#在outpth目录下创建ndir路径中的所有目录，是否使用决对路径
def makeDir(outpth,ndir):
    tmpdir = ''
    if ndir[0] == '/':
        tmpdir = outpth + ndir
    else:
        tmpdir = outpth + '/' + ndir
    print tmpdir
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

# 创建一个目录下的所有子目录到另一个目录
def createDirs(spth,tpth):
    files = getAllExtFile(spth,'.*')
    makedirstmp = []
    isOK = True
    # 分析所有要创建的目录
    tmpfpth = fpth
    for d in files:
        if d[1] != '/' and (not d[1] in makedirstmp): #创建未创建的目录层级
            tmpdir = d[1][1:]
            tmpleves = tmpdir.split('/')
            alldirs = getAllLevelDirs(tmpleves)
            for dtmp in alldirs:
                if not dtmp in makedirstmp:
                    makeDir(tpth,dtmp)
                    makedirstmp.append(dtmp)

# 替换文件名
def replaceFileName(path,sname,replaceStr,tostr):
    a = sname
    tmpname = a.replace(replaceStr, tostr)
    outpath = path + tmpname
    oldpath = path + sname
    cmd = "mv %s %s"%(oldpath,outpath)
    print cmd
    os.system("mv %s %s"%(oldpath,outpath))

# 替换目录下的文件名中某个字符串为其他字符串
def renameDir(sdir,replacestr,tostr,exittype):
    files = getAllExtFile(sdir,fromatx = exittype)
    allfilepath = []
    for f in files:
        tmppath = sdir + f[1]
        filename = f[2] + exittype
        allfilepath.append([tmppath,filename])
    for p in allfilepath:
        replaceFileName(p[0], p[1], replacestr, tostr)


def conventFileName(fname):
    tmp = fname.split('/')[-1]
    fname = tmp.split('.')[0]
    k = timetool.conventDayStrAdd_(fname)
    return k


def downCmdRun(days):

    if not days:
        print 'all data is downloaded...'
        return
    cmd = '/bin/sh /Users/mage/Documents/btc/klineokex/downkline.sh '

    for d in days:
        tmpcmd = cmd + d + '.txt'
        os.system(tmpcmd)


def downAllKlineDataFromServer():

    files = getAllFiles('out','.txt')
    print files
    klines = []
    lasttime = 0
    sntimes = []

    tmpfiles = {}

    for p in files:
        k = conventFileName(p)
        tmpfiles[k] = p
    fs = tmpfiles.keys()
    fs.sort(reverse = False)

    lastday = fs[-1].split('/')[-1].split('.')[0]

    print lastday

    days = timetool.getDateDaysFromOneDate(lastday)
    downdays = days[:-1]
    downCmdRun(downdays)

def main():

    downAllKlineDataFromServer()

    files = getAllFiles('out','.txt')
    print files
    klines = []
    lasttime = 0
    sntimes = []

    tmpfiles = {}

    for p in files:
        k = conventFileName(p)
        tmpfiles[k] = p
    fs = tmpfiles.keys()
    fs.sort(reverse = False)

    for k in fs:
        p = tmpfiles[k]
        tpth = 'out' + p
        f = open(tpth,'r')
        tmpk = f.read()
        f.close()
        klinetmps = json.loads(tmpk)
        #[1510588800000,65.146,65.146,64.966,65.012,5616.0,863.44491422901]
        #时间戳，开，高，低，收，交易量，交易量转化为BTC或LTC数量
        # print len(klinetmps)
        
        sntimes.append([klinetmps[0][0],klinetmps[-1][0]])
        startn = 0
        for n in range(len(klinetmps)):
            if lasttime == klinetmps[n][0]:
                startn = n + 1
        if klines:
            print klines[-1],klinetmps[startn-1]
        klines += klinetmps[startn:-1]
        lasttime = klines[-1][0]
        
    print len(klines)
    # print sntimes
#测试
if __name__ == '__main__':
    args = sys.argv
    fpth = ''
    if len(args) == 2 :
        if os.path.exists(args[1]):
            fpth = args[1]
        else:
            print "请加上要转码的文件路径"
    else:
        print "请加上要转码的文件路径"
        main()
    

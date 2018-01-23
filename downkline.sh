#!bin/bash
#create buy zhangjunpeng @ 2016
#export PATH=/Users/junpengzhang/Documents/android/apktool:
#安卓sdk
#export PATH=/Users/junpengzhang/Documents/android/android_sdk/build-tools/25.0.0:
export PATH=/usr/bin/:/usr/local/bin:/bin:

#获取当前目录的命令是pwd
#获取脚本所在目录是${cd `dirname `; pwd},把{换成括号，模版里不识别括号
#运行程序，并保存pid
#获取日期和时间
#DATE=`date "+%Y-%m-%d-%H:%M:%S"`

# DATE=`date "+%Y-%m-%d %H:%M:%S"`
# echo $DATE

# svn up

# LOG=`nohup python btc38urlhttp.py > log.txt 2>&1 & echo $!`
# # LOG="12345"
# echo $LOG
# OUTSTR=$DATE"\n"$LOG
# echo $OUTSTR > psid.txt

if [[ $1 ]]; then
    #statements
    scp root@btc.woodcol.com:/home/woodcol/btcctrade/test/nnbtc/nnbtc/out/${1} /Users/mage/Documents/btc/klineokex/out/${1}

else
    echo "请输入要下载的文件名"

fi



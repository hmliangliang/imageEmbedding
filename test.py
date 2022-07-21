#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 15:10
# @Author  : Liangliang
# @File    : test.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow import keras
import requests
import os
import time
import datetime
import s3fs
import math
import argparse
import pandas as pd
import base64
from multiprocessing.dummy import Pool
from sklearn.decomposition import PCA

result = 0

def multiprocessingWrite(file_number,data,output_path,count):
    #print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    #s3fs.S3FileSystem = S3FileSystemPatched
    #fs = s3fs.S3FileSystem()
    with open(os.path.join(output_path, 'pred_{}_{}.csv'.format(count,int(file_number))), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个大数据文件的第{}个子文件已经写入完成,写入数据的行数{} {}".format(count,file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output


def write(data, args, count):
    #注意在此业务中data是一个二维list
    n_data = len(data) #数据的数量
    n = math.ceil(n_data/args.file_max_num) #列表的长度
    start = time.time()
    for i in range(0,n):
        multiprocessingWrite(i, data[i * args.file_max_num:min((i + 1) * args.file_max_num, n_data)],
                                 args.data_output, count)
    cost = time.time() - start
    print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))

def getEmbedding(roleid, i, data_url, count):
    if i%200000 == 0:
        print("开始计算第{}个文件的第{}个样本的embedding!".format(count,i))
    #将url地址从base64编码进行解码
    img_url = str(base64.b64decode(data_url), 'utf-8')
    #下载数据
    r = requests.get(img_url, stream=True)
    #返回状态码
    if r.status_code == 200:
        with open('tmp.png', 'wb') as f:
            f.write(r.content)  # 将内容写入图片
        img = tf.io.read_file('tmp.png')
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, (1, 224, 224, 3))
        img = tf.keras.applications.densenet.preprocess_input(img)
        try:
            embed = model(img)
            global result
            result[i, 0] = str(roleid)
            result[i, 1] = data_url
            result[i, 2::] = embed.numpy().astype("str")
        except Exception as e:
            pass
    else:
        print("第{}个样本URL读取异常!".format(i))
    if i%200000 == 0:
        print("第{}个文件的第{}个样本的embedding计算完成!".format(count,i))

if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--dim", help="数据的恶输出维数", type=int, default=100)
    parser.add_argument("--thread_num", help="多线程编程的线程数目", type=int, default=1000)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=800000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()

    # 读取数据文件
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0

    #读取模型
    model = tf.keras.applications.densenet.DenseNet201(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling="avg",
    classes=1000) #注: tensorflow2.3没有参数classifier_activation="softmax"
    #获取最后一个中间层的输出结果
    model = keras.models.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    '''
    Layer (type)                       Output Shape         Param #        Connected to
    avg_pool (GlobalAveragePooling 2D) (None, 1920)           0           ['relu[0][0]']
    '''
    #读取图像数据
    for file in input_files:
        pool = Pool(processes=args.thread_num)
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.read_csv("s3://" + file, sep=',', header=None, usecols=[0, 1]).astype('str')  # 读取数据,第一列为id,第二列为中文txt
        n = data.shape[0]
        result = np.zeros((n, 1922)).astype("str") #roleid + url + 1920维特征
        for i in range(n):
            pool.apply_async(func=getEmbedding, args=(data.iloc[i, 0], i, data.iloc[i, 1], count,))
        pool.close()
        pool.join()
        #删选出roleid非零的行,roleid为0表示该行对应的玩家没有图片或无法获得image embedding
        result = result[np.argwhere(result[:, 0] > "0.0")[:, 0], :]
        #输出的数据维度为1920维,进行降维
        pca = PCA(n_components=args.dim)
        result[:,2:102] = pca.fit_transform(result[:,2::].astype(np.float32)).astype("str")
        result = result[:,0:102]
        write(result.tolist(), args, count)
    print("已完成第{}个文件数据的推断! {}".format(count, datetime.datetime.now()))
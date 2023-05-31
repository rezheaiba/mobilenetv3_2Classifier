"""
# @Time    : 2022/8/3 14:12
# @File    : 批量改名.py
# @Author  : rezheaiba
"""
import os
import shutil
import random

import cv2
import numpy as np
from PIL import Image


def rename(folder):
    folder_name = folder
    # h:当前进入的路径 d:当前路径下的dir f:当前路径下的files
    for h, d, f in os.walk(folder_name):
        id = 1
        # file_names = os.listdir(h)
        os.chdir(h)
        for name in f:
            if f is None:
                break
            os.rename(name, str('%08d' % id) + '.jpg')
            id += 1


def rename_folder(string):
    folder_name = string
    # h:当前进入的路径 d:当前路径下的dir f:当前路径下的files
    for h, d, f in os.walk(folder_name):
        id = 1
        # file_names = os.listdir(h)
        os.chdir(h)
        for name in f:
            if f is None:
                break
            os.rename(name, str(str(os.path.splitext(h)[0]).split('\\')[-1]) + '_' + str('%05d' % id) + '.jpg')
            id += 1


def delWithStr(folder, delstr):
    folder_name = folder
    # h:当前进入的路径 d:当前路径下的dir f:当前路径下的files
    for h, d, f in os.walk(folder_name):
        os.chdir(h)
        for name in f:
            if f is None:
                break
            if name.__contains__(delstr):
                os.remove(name)


# 位深变为24
def redtype(path="D:\Dataset\data\level1\hdr", save_path="D:\Dataset\data\level1\hdr"):
    path = path
    save_path = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    files = os.listdir(path)
    # print(files)

    for pic in files:
        img = Image.open(os.path.join(path, pic)).convert('RGB')
        print(img.getbands())  # ('P',) 这种是有彩色的，而L是没有彩色的
        pic_new = os.path.join(save_path, pic)

        img.save(pic_new)


# 位深变为24
def redtypeAll(folder_name):
    for h, d, f in os.walk(folder_name):
        os.chdir(h)
        for name in f:
            img = Image.open(os.path.join(h, name)).convert('RGB')
            print(img.getbands())  # ('P',) 这种是有彩色的，而L是没有彩色的
            pic_new = os.path.join(h, name)
            print(pic_new)
            img.save(pic_new)


def rgb2gray(path=r"F:\Dataset\rainy", save_path=r"F:\Dataset\rainy_L"):
    path = path
    save_path = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    files = os.listdir(path)

    for pic in files:
        img = Image.open(os.path.join(path, pic)).convert('L')
        print(img.getbands())  # ('P',) 这种是有彩色的，而L是没有彩色的
        pic_new = os.path.join(save_path, pic)

        img.save(pic_new)


# 有很多相似数据集排列在一起，打乱他们
def random_name(folder):
    folder_name = folder
    for h, d, f in os.walk(folder_name):
        os.chdir(h)
        try:
            arange_f = np.arange(len(f))
            np.random.shuffle(arange_f)
            print(arange_f)
        except:
            # 在这里并不会用到，f可以为0
            print('{f}为空')
        for index in range(len(f)):
            if f is None:
                break
            # 打乱文件名字
            os.rename(f[index], str(arange_f[index]) + '.jpg')


def MyTest():
    # img = Image.open('rain.jpg').convert('L')
    # img.save('1.jpg')
    # print(img.shape)
    img = Image.open('../001-1.jpg').convert('RGB')
    print(img.getbands())  # ('P',) 这种是有彩色的，而L是没有彩色的
    print(np.shape(img))
    img.save('001-2.jpg')


def split(root_path, num=300):
    output_path_train = os.path.join(root_path + str(num))
    for h, d, f in os.walk(root_path):
        if len(f) == 0:
            continue
        f_train = random.sample(f, num)  # sample(seq, n) 从序列seq中选择n个随机且独立的元素
        # h是当前遍历的目录路径
        output_dir_train = h.replace(root_path, output_path_train)
        os.makedirs(output_dir_train, exist_ok=True)
        a = 0
        lentrain = len(f_train)
        for fts in f_train:
            ori_path = os.path.join(h, fts)  # 该图像的文件名称路径
            new_path = os.path.join(output_dir_train, fts)  # 新路径的文件地址
            shutil.copyfile(ori_path, new_path)  # ori复制到new，其中new必须是完整的目标文件名
            a += 1
            print(h, ": "
                  , a, lentrain)


if __name__ == '__main__':
    # rename(folder=r'D:\Dataset\data\test_dataset\夜晚图')  # 对一个文件夹下深度遍历后的所有文件
    # random_name(folder = r'D:\Dataset\data\final-450-gray')  # 打乱文件夹中图片顺序
    rename_folder(r'D:\Dataset\data\test_dataset\夜晚图')  # 根据文件夹名称对图片命名

    # rgb2gray(path=r"D:\Dataset\data\genrain\c_", save_path=r"D:\Dataset\data\genrain\c_")  # 彩转黑
    # redtype(path=r"D:\Dataset\places365_standard\val\tree_farm", save_path=r"D:\Dataset\places365_standard\val\tree_farm")  # 把位深改为为24

    # redtypeAll(folder_name=r'D:\Dataset\data\level1_aug_splited\train\wdr')

    # delWithStr(folder=r'D:\Dataset\data\level1\stripe', delstr='副本')  # 根据名称批量删除图片
    # MyTest()
    # split(root_path=r'D:\Dataset\data\final\train\indoor', num=3000)  # 随机导出num张图片

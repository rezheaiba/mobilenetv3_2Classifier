"""
# @Time    : 2022/8/1 9:49
# @File    : make_indoor67.py
# @Author  : rezheaiba
"""
import os, shutil, random

from tqdm import tqdm


def split(root_path, train_test_rate=[9, 1]):
    output_path_train = os.path.join(root_path + "_splited", "train")
    output_path_test = os.path.join(root_path + "_splited", "test")
    train_rate = train_test_rate[0]
    test_rate = train_test_rate[1]
    for h, d, f in os.walk(root_path):
        lenf = len(f)  # files:除去root_path目录下的文件夹之外的files
        f_train = random.sample(f, int(len(f) // (
                    (train_rate + test_rate) / train_rate)))  # sample(seq, n) 从序列seq中选择n个随机且独立的元素
        f_test = f[:]
        # h是当前遍历的目录路径
        output_dir_train = h.replace(root_path, output_path_train)
        output_dir_test = h.replace(root_path, output_path_test)
        os.makedirs(output_dir_train, exist_ok=True)
        os.makedirs(output_dir_test, exist_ok=True)
        a = 0
        lentrain = len(f_train)
        for fts in f_train:
            f_test.remove(fts)
            ori_path = os.path.join(h, fts)  # 该图像的文件名称路径
            new_path = os.path.join(output_dir_train, fts)  # 新路径的文件地址
            shutil.copyfile(ori_path, new_path)  # ori复制到new，其中new必须是完整的目标文件名
            a += 1
            print(h, ": ", a, lentrain)
        a = 0
        lentest = len(f_test)
        for fts in f_test:
            ori_path = os.path.join(h, fts)
            new_path = os.path.join(output_dir_test, fts)
            shutil.copyfile(ori_path, new_path)
            a += 1
            print(h, ": ", a, lentest)


if __name__ == '__main__':
    split(r'D:\Dataset\data\level1-450_plus', train_test_rate=[10,1])

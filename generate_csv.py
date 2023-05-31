"""
# @Time    : 2023/2/7 11:17
# @File    : 生成csv.py
# @Author  : rezheaiba
"""
import csv
import os
import random

os.makedirs('label', exist_ok=True)
f_train = open('label/train_label.csv', 'w', encoding='utf-8', newline='')
f_test = open('label/test_label.csv', 'w', encoding='utf-8', newline='')
csv_writer_train = csv.writer(f_train)
csv_writer_test = csv.writer(f_test)


if __name__ == '__main__':
    # 构建表头
    # fileHeader = ["path", "label1", "label2", "label3", "label4"]
    # csv_writer_train.writerow(fileHeader)
    # csv_writer_test.writerow(fileHeader)

    # get the path of the train_dataset
    root = '../datasets/final8_mult/train'
    filelist = os.listdir(root)
    
    train_list = []
    for path in filelist:
        current_path = os.path.join(root, path)
        train_list.extend(os.listdir(current_path))
        
    random.shuffle(train_list)
    for trainName in train_list:
        # label1:0-indoor 1-outdoor
        # label2:0-flying-dust 1-fog 2-rainy 3-wdr 4-other
        # label3:0-stronglight 1-other
        # label4:0-stripe 1-other
        if 'flying-dust' in trainName:
            continue
        if 'indoor' in trainName:
            label = '5'
        if 'outdoor' in trainName:
            label = '5'
        if 'fog' in trainName:
            label = '0'
        if 'rainy' in trainName:
            label = '1'
        if 'rainy-light' in trainName:
            label = '1'
        if 'wdr' in trainName:
            label = '2'
        if 'stronglight' in trainName:
            label = '3'
        if 'stronglight-indoor' in trainName:
            label = '3'
        if 'stripe' in trainName:
            label = '4'
        csv_writer_train.writerow([trainName, label])

    # get the path of the test_dataset
    root = '../datasets/final8_mult/test'
    filelist = os.listdir(root)
    
    test_list = []
    for path in filelist:
        current_path = os.path.join(root, path)
        test_list.extend(os.listdir(current_path))
         
    random.shuffle(test_list)
    for testName in test_list:
        if 'flying-dust' in testName:
            continue
        if 'indoor' in testName:
            label = '5'
        if 'outdoor' in testName:
            label = '5'
        if 'fog' in testName:
            label = '0'
        if 'rainy' in testName:
            label = '1'
        if 'wdr' in testName:
            label = '2'
        if 'stronglight' in testName:
            label = '3'
        if 'stripe' in testName:
            label = '4'
        csv_writer_test.writerow([testName, label])
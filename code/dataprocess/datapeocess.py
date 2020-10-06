import os
import json
import argparse  # python用于解析命令行参数和选项的标准模块，类似于linux中的ls指令
import skimage
import skimage.transform
import skimage.io
import torch
import gc
from PIL import Image

data_path_text = '../../data/text/'  # 文本数据集路径
data_path_annotation = '../../data/image/annotations/'  # 图片注释数据集路径
data_path_image = '../../data/image/val2017/'  # 图片数据集路径


def processtext(data_path=data_path_text):
    datasets = ['val.txt']  # , 'val.txt'], 'test.txt']  # 数据集名称
    sum_character = 0  # 总的字符数
    sum_data = 0  # 总的文章数
    new_data = []  # 新的数据，每个列表元素代表一篇文章
    for data in datasets:
        filepath = os.path.join(data_path, data)
        newdata = []
        print('Start process textdata :' + filepath)
        with open(filepath, 'r', encoding='utf-8') as fr:
            file = fr.readlines()
        for filerow in file:
            filerow = filerow.replace('<s>', '')
            filerow = filerow.replace('<s>', '')
            filerow = filerow.replace('</s>', '')
            filerow = filerow.replace('\n', '')
            newdata.append([filerow])  # 将处理好的数据添加进列表中
            sum_data += 1
        new_data.append(newdata)
        print('The text process is over' + filepath)
        fr.close()  # 关闭文件
    print('文本总数据：', sum_data)
    return new_data  # 返回处理好后的数据


def processimageannotations(data_path=data_path_annotation):
    datasets = ['captions_val2017.json']  # ,'captions_val2017.json']
    new_data = []  # 处理完的新数据
    newdata = []
    for data in datasets:
        filepath = os.path.join(data_path, data)
        print('Start process imageannotation:' + filepath)
        with open(filepath, 'r', encoding='utf-8') as fr:
            data_list = json.load(fr)  # 以json的方式加载数据
        ff = open(data_path + 'processcaptionsval2017.json', 'w', encoding='utf-8')
        images = data_list['images']
        annotations = data_list['annotations']
        image_list = []  # 只包含图片ID和图片名称的列表
        anotation_list = []  # 只包含图片ID和说明文字的列表
        image_data = 0  # 图片总数
        anotation_data = 0  # 注释总数

        for image in images:  # 得到图片的ID和图片名称
            image_list.append([image['id'], image['file_name'], image['height'], image['width']])
            image_data += 1
        for annotation in annotations:  # 得到对应图片ID和说明文字
            anotation_list.append([annotation['image_id'], annotation['caption']])
            anotation_data += 1
        number = 0
        for i in image_list:  # 匹配对应的图片所有的caption
            caption = []
            for j in anotation_list:
                if (i[0] == j[0]):
                    caption.append(j[1])
                    anotation_list.remove(j)
                    number += 1
                    print(number)
            all_data = i + caption
            newdata.append(all_data)
            wenben = {'annotation': all_data}
            wenben = json.dumps(wenben)
            ff.write(wenben)
            ff.write('\n')
        new_data.append(newdata)
        fr.close()  # 关闭文件
        ff.close()
        print("The imageannotation process is over" + filepath)
    return new_data


def processimage(data_path=data_path_image):  # 图片处理函数
    file_path = data_path + '/*.jpg'
    print('Start process imagedata:' + file_path)
    datasets = skimage.io.ImageCollection(file_path)  # 加载图片数据集
    file = []  # 文件名
    data = []  # 图片的array
    jishu = 0
    for filename in os.listdir(data_path):  # 获取图片文件名
        file.append(filename)
    for i in datasets:
        i = i.transpose()
        i = skimage.transform.resize(i, (3,224,224))
        data.append(i)
        jishu += 1
        print(jishu)
    print('The image process is over' + file_path)
    return data, file


def processimage2(data_path=data_path_image):  # 图片处理函数
    data = []
    file = []
    for filename in os.listdir(data_path):
        image = Image.open(data_path + filename)
        print(image.size)
        file.append(filename)
        data.append(image)
    return data, file

def processimage3(data_path = data_path_image):
    print('Start process imagedata:')
    file = []  # 文件名
    data = []  # 图片的array
    jishu = 0
    for filename in os.listdir(data_path):  # 获取图片文件名
        image = skimage.io.imread(data_path+filename)
        new_image = skimage.transform.resize(image, (3, 224, 224))
        file.append(filename)
        data.append(new_image)
        jishu+=1
        print(jishu)
    print('The image process is over')
    return data, file

def processdata():  # 数据处理函数
    file_path = '../../data/image/annotations/processcaptionsval2017.json'
    annotation = []
    print('Start process text  ' + file_path)
    with open(file_path, 'r', encoding='utf-8') as file:  # 获得验证集的annotation
        for i in file:
            data = json.loads(i)
            annotation.append(data['annotation'])
    file.close()
    print('The textprocess is over  ' + file_path)
    image, filename = processimage()  # 获取图片和文件名称
    imagedata = []
    for i in range(5000):
        imagedata.append([filename[i], image[i]])
    for j in annotation:
        for k in imagedata:
            if (j[1] == k[0]):  # 图片名称一样则匹配成功
                j.append(k[1])
                break
    text = processtext()
    image_length = len(image)  # 图片的数量
    add = 0
    for k in range(len(text[0])):  # 将文本与图片信息结合,得到最终数据集
        if (k % (image_length - 2) == 0):
            add = 0
        text[0][k] += annotation[add][4:-1]
        text[0][k] += annotation[add + 1][4:-1]
        text[0][k] += annotation[add + 2][4:-1]
        text[0][k].append(annotation[add][-1])
        text[0][k].append(annotation[add + 1][-1])
        text[0][k].append(annotation[add + 2][-1])
        add += 1
    gc.collect()  # 回收内存
    return text  # 返回最终处理好的数据
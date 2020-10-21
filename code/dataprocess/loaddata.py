import os
import torch
import numpy
import json
import skimage
import skimage.transform
import skimage.io
import skimage.color
from tqdm import tqdm  # 显示进度条

class Load_data:

    def __init__(self, data_path_text, data_path_annotation, data_path_image, datasize):
        self.data_path_text = data_path_text  # 文本集文件路径
        self.data_path_annotation = data_path_annotation  # 图片注释文件路径
        self.data_path_image = data_path_image  # 图片文件路径
        self.datasize = datasize

    def load_text(self):  # 得到每篇文章
        data_path = self.data_path_text
        datasets_text = ['val.txt', 'test.txt']  # , 'val.txt'], 'test.txt']en_core_web_md  # 数据集名称
        text_data = []  # 每个列表元素代表一篇文章
        for data in datasets_text:
            filepath = os.path.join(data_path, data)
            newdata = []
            print('*' * 80)
            print('Start process textdata :' + os.path.realpath(filepath))
            datasize = 0
            with open(filepath, 'r', encoding='utf-8') as f_t:
                file = f_t.readlines()
            for filerow in tqdm(file):  # 实时显示进度
                if( datasize == self.datasize ):    # 控制数据集大小
                    break
                filerow = filerow.replace('<s>', '')
                filerow = filerow.replace('<s>', '')
                filerow = filerow.replace('</s>', '')
                filerow = filerow.replace('\n', '')
                newdata.append(filerow)  # 将处理好的数据添加进列表中
                datasize += 1
                pass
            text_data.append(newdata)
            print('The number of total text: ', len(newdata))
            print('*' * 80, '\n')
            f_t.close()  # 关闭文件
        return text_data

    def load_sentence(self):
        data_path = self.data_path_text
        datasets_sentence = ['val_sentence.json', 'test_sentence.json']  # 句子数据集
        sentence_data = []  # 每个列表元素代表一篇文章的分句
        for data in datasets_sentence:
            filepath = os.path.join(data_path, data)
            newdata = []
            print('*' * 80)
            print('Start process sentencedata :' + os.path.realpath(filepath))
            datasize = 0
            sum_sentence = 0  # 记录总句数
            with open(filepath, 'r', encoding='utf-8') as f_s:
                file = f_s.readlines()
            for filerow in tqdm(file):  # 实时显示进度
                if (datasize == self.datasize):
                    break
                sentnce = json.loads(filerow)
                newdata.append(sentnce['sentences'])  # 将处理好的数据添加进列表中
                sum_sentence += len(sentnce['sentences'])
                datasize += 1
                pass
            sentence_data.append(newdata)
            print('The number of total sentence: ', sum_sentence)
            print('*' * 80, '\n')
            f_s.close()  # 关闭文件
        return sentence_data

    def load_annotation(self):
        data_path = self.data_path_annotation
        datasets = ['processed_captions_val2017.json', 'processed_captions_train2017.json']
        annotation_data = []
        for data in datasets:
            file_path = os.path.join(data_path, data)
            newdata = []
            print('*' * 80)
            print('Start process text  ' + os.path.realpath(file_path))
            with open(file_path, 'r', encoding='utf-8') as f_a:  # 获得验证集的annotation
                file = f_a.readlines()
            for filerow in tqdm(file):
                annotation = json.loads(filerow)
                newdata.append(annotation['annotation'])
                pass
            annotation_data.append(newdata)
            print('*' * 80, '\n')
            f_a.close()
        return annotation_data

    def load_image(self):
        data_path = self.data_path_image
        datasets = ['new_val2017', 'new_test2017']
        image_data = []
        for data in datasets:
            filepath = data_path + data
            datasize = 0
            print('*' * 80)
            print('Start process imagedata:' + os.path.realpath(filepath))
            new_data = []
            list_file = os.listdir(filepath)
            for filename in tqdm(list_file):
                if datasize == self.datasize:
                    break
                image = skimage.io.imread(filepath + '/' + filename)
                if (image.shape[-1] != 3):    # 将灰度图转为彩色图
                    image = skimage.color.gray2rgb(image)
                newimage = torch.tensor(numpy.transpose(image, (2, 0, 1)), dtype=torch.float)
                new_data.append(newimage)
                datasize += 1
            image_data.append(new_data)
            print('*' * 80, '\n')
        return image_data

    def load_dependencytree(self):
        data_path = self.data_path_text
        datasets = ['val_parents.json', 'test_parents.json']
        dependency_data = []
        for data in datasets:
            filepath = os.path.join(data_path, data)
            newdata = []
            datasize = 0
            print('*' * 80)
            print('Start process dependency sentence:' + os.path.realpath(filepath))
            with open(filepath, 'r', encoding='utf-8') as f_d:
                file = f_d.readlines()
            for parents in tqdm(file):
                if datasize == self.datasize:
                    break
                parent = json.loads(parents)
                newdata.append(parent['parent'])
                datasize += 1
                pass
            dependency_data.append(newdata)
            print('*' * 80, '\n')
            f_d.close()
        return dependency_data

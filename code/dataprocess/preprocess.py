import os
import json
import spacy
from PIL import Image
from tqdm import tqdm  # 显示进度条


class preprocess_data:

    def __init__(self, data_path_text, data_path_annotation, data_path_image, mini_character, spacy_model):
        self.data_path_text = data_path_text  # 文本集文件路径
        self.data_path_annotation = data_path_annotation  # 图片注释文件路径
        self.data_path_image = data_path_image  # 图片文件路径
        self.mini_character = mini_character  # 小于此长度的sentence不进行句法分析
        self.NLP_spacy = spacy.load(spacy_model)  # 加载spacy的语言模型

    def get_Dependencytree(self, sentence, savefile):  # 构造依存树
        file = open(savefile, 'a', encoding='utf-8')
        parents = []
        for sents in sentence.sents:
            if len(sents) > self.mini_character:
                child = []
                for token in sents:
                    parent = token.head
                    position = 0
                    if token.text != parent.text:   # 说明不是根结点
                        for index in sents:  # 查找parent的位置
                            if index.text == parent.text:
                                position += 1
                                break
                            else:
                                position += 1
                    child.append(position)
                parents.append(child)
        file.write(json.dumps({'parent': parents}))
        file.write('\n')
        file.close()

    def get_sentence(self):  # 将文本使用spacy分句
        data_path = self.data_path_text
        datasets = ['val.txt', 'test.txt']  # 数据集名称
        for data in datasets:
            filepath = os.path.join(data_path, data)
            file_firstname = data.replace('.txt', '')  # 得到文件的标识名
            save_sentence = data_path + file_firstname + '_sentence.json'  # 保存的文件名
            if os.path.exists(save_sentence) == True:  # 判断文件是否存在
                print('The ' + file_firstname + '_sentence.json' + ' is existed', '\n')
            else:
                print('*' * 80)
                print('Start save sentences :' + os.path.realpath(save_sentence))
                with open(filepath, 'r', encoding='utf-8') as fr:
                    file = fr.readlines()
                for filerow in tqdm(file):
                    filerow = filerow.replace('<s>', '')
                    filerow = filerow.replace('<s>', '')
                    filerow = filerow.replace('</s>', '')
                    filerow = filerow.replace('\n', '')
                    sentence = self.NLP_spacy(filerow)  # spacy
                    # 进行依存分析
                    self.get_Dependencytree(sentence, data_path + file_firstname + '_parents.json')

                    all_sentence = [str(_) for _ in sentence.sents]
                    all_sentence = {'sentences': all_sentence}
                    all_sentence = json.dumps(all_sentence)
                    with open(save_sentence, 'a') as sentence_file:  # 写入句子信息
                        sentence_file.write(all_sentence)
                        sentence_file.write('\n')
                    pass
                print('*' * 80, '\n')
                fr.close()  # 关闭文件
                sentence_file.close()  # 关闭句子文件

    def get_imageannotations(self):  # 预处理image annotation, 将image与annotation相对应
        data_path = self.data_path_annotation
        datasets = ['captions_val2017.json'] #, 'captions_train2017.json']
        for data in datasets:
            newdata = []
            filepath = os.path.join(data_path, data)
            save_annotation = data_path + 'processed_' + data  # 保存的文件名
            if os.path.exists(save_annotation) == True:  # 判断文件是否存在
                print('The ' + 'processed_' + data + ' is existed', '\n')
            else:
                f_a = open(save_annotation, 'w', encoding='utf-8')
                print('*' * 80)
                print('Start process image annotation:' + os.path.realpath(save_annotation))
                with open(filepath, 'r', encoding='utf-8') as f_r:
                    data_list = json.load(f_r)  # 以json的方式加载数据
                images = data_list['images']
                annotations = data_list['annotations']
                image_list = []  # 只包含图片ID和图片名称的列表
                anotation_list = []  # 只包含图片ID和说明文字的列表
                for image in images:  # 得到图片的ID和图片名称
                    image_list.append([image['id'], image['file_name'], image['height'], image['width']])
                for annotation in annotations:  # 得到对应图片ID和说明文字
                    anotation_list.append([annotation['image_id'], annotation['caption']])
                for i in tqdm(image_list):  # 匹配对应的图片所有的caption
                    caption = []
                    for j in anotation_list:
                        if (i[0] == j[0]):
                            caption.append(j[1])
                            anotation_list.remove(j)
                    all_data = i + caption
                    newdata.append(all_data)
                    wenben = {'annotation': all_data}
                    wenben = json.dumps(wenben)
                    f_a.write(wenben)
                    f_a.write('\n')
                    pass
                f_r.close()  # 关闭文件
                f_a.close()
                print('*' * 80, '\n')

    def resize_image(self):  # 将图片全部变为224 224大小
        data_path = self.data_path_image
        datasets = ['val2017']  #
        for data in datasets:
            filepath = os.path.join(data_path, data)
            save_image = data_path + 'new_' + data + '/'
            if os.path.exists(save_image) == True:  # 判断文件是否存在
                print('The ' + 'new_' + data + '/' + ' is existed', '\n')
            else:
                os.mkdir(save_image)  # 创建文件夹
                print('*' * 80)
                print('Start resize image: ' + os.path.realpath(save_image))
                list_file = os.listdir(filepath)
                for filename in tqdm(list_file):  # 获取图片文件名
                    image = Image.open(filepath + '/' + filename)
                    new_image = image.resize((224, 224), Image.ANTIALIAS)
                    new_image.save(save_image + filename)
                    pass
                print('*' * 80, '\n')



"""
parser = config.parse_args()

pre = preprocess_data(parser.text_path, parser.annotation_path, parser.image_path, parser.mini_character,
                      parser.spacy_model)

a = pre.get_sentence()
# b = pre.get_imageannotations()
# c = pre.get_sentence()
"""
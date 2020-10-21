import os
import zipfile

text_path = '../data/text/'
image_path = '../data/image/'
annotation_path = '../data/image/annotations/'


def unzip_text(filepath=text_path):
    datasets = ['val.zip', 'val_sentence.zip', 'val_parents.zip']
    jishu = 0
    for data in datasets:
        try:
            file = zipfile.ZipFile(filepath + data)
            if (jishu == 0):
                dirname = data.replace('.zip', '.txt')
            else:
                dirname = data.replace('.zip', '.json')
            # 如果存在与压缩包同名文件夹 提示信息并跳过
            if os.path.exists(filepath + dirname):
                print(f'{os.path.realpath(filepath + dirname)} dir has already existed', '\n')
                jishu += 1
            else:
                file.extractall(filepath)
                file.close()
                print('The ' + os.path.realpath(filepath + dirname) + ' unzip successfully', '\n')
                jishu += 1
        except:
            print(f'{os.path.realpath(filepath + data)} unzip fail', '\n')
            jishu += 1


def unzip_image(filepath=image_path):
    datasets = ['new_val2017.zip']
    for data in datasets:
        try:
            file = zipfile.ZipFile(filepath + data)
            dirname = data.replace('.zip', '')
            # 如果存在与压缩包同名文件夹 提示信息并跳过
            if os.path.exists(filepath + dirname):
                print(f'{os.path.realpath(filepath + dirname)} dir has already existed', '\n')
            else:
                file.extractall(filepath)
                file.close()
                print('The ' + os.path.realpath(filepath + dirname) + ' unzip successfully', '\n')
        except:
            print(f'{os.path.realpath(filepath + data)} unzip fail', '\n')


def unzip_annotation(filepath=annotation_path):
    datasets = ['processed_captions_val2017.zip', 'processed_captions_train2017.zip']
    for data in datasets:
        try:
            file = zipfile.ZipFile(filepath + data)
            dirname = data.replace('.zip', '.json')
            # 如果存在与压缩包同名文件夹 提示信息并跳过
            if os.path.exists(filepath + dirname):
                print(f'{os.path.realpath(filepath + dirname)} dir has already existed', '\n')
            else:
                file.extractall(filepath)
                file.close()
                print('The ' + os.path.realpath(filepath + dirname) + ' unzip successfully', '\n')
        except:
            print(f'{os.path.realpath(filepath + data)} unzip fail', '\n')

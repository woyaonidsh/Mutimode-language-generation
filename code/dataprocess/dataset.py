from tqdm import tqdm
from copy import copy, deepcopy

import torch
import torch.utils.data as data
from model.CreateTree import CreateTree


# Dataset class for Mutimodel dataset
class MutimodelDataset(data.Dataset):
    def __init__(self, loaddata, tokenizer, image_number, make_batch):  # argus.image_number):
        super(MutimodelDataset, self).__init__()
        self.image_number = image_number  # 表示图片与文本匹配的数量
        self.loaddata = loaddata  # 加载数据的类
        self.tokenizer = tokenizer  # BERT编码器
        self.batch = make_batch  # Transformer的batch函数

        self.size = []  # 数据集大小
        self.documents, self.src_masks, self.tgt_masks = self.read_document()  # 每篇文章变为编码ID
        self.images = self.read_image()  # 每篇文章的图片

        self.sentences, self.trees = self.read_sentences()  # 每句话变为编码ID
        self.targets = self.read_labels()  # 得到标签

    def __getitem__(self, index):  # 可以让对象具有迭代功能
        documents = copy(self.documents[index])
        src_masks = copy(self.src_masks[index])
        sentences = copy(self.sentences[index])
        images = copy(self.images[index])
        trees = copy(self.trees[index])
        targets = copy(self.targets[index])
        tgt_masks = copy(self.tgt_masks[index])
        return documents, src_masks, sentences, images, trees, targets, tgt_masks  # 返回最终数据

    def read_sentences(self):
        sentences = self.loaddata.load_sentence()
        parents = self.loaddata.load_dependencytree()
        trees = self.read_trees()
        sentence_dataset = []
        tree_dataset = []
        print('read_sentence: now')
        for sentence in range(len(sentences)):  # 每个数据集
            data = []
            tree_data = []
            for sent in tqdm(range(len(sentences[sentence]))):  # 每篇文章
                document = []
                tree = []
                for every_sentence in range(len(sentences[sentence][sent])):
                    token = self.tokenizer.encode(sentences[sentence][sent][every_sentence])
                    if len(token) >= len(parents[sentence][sent][every_sentence]):
                        document.append(torch.tensor(token))  # 满足条件的句子被选中,句子长度 > parent长度
                        tree.append(trees[sentence][sent][every_sentence])
                data.append(document)
                tree_data.append(tree)
                pass
            sentence_dataset.append(data)  # 句子的数据集
            tree_dataset.append(tree_data)   # 树的数据集
        return sentence_dataset, tree_dataset  # 返回文本数据集

    def read_document(self):
        document_data = self.loaddata.load_text()
        document_dataset = []
        document_src = []
        document_tgt = []
        print('read_document: now')
        for dataset in document_data:  # 数据集
            data = []
            document_src_mask = []
            document_tgt_mask = []
            for document in tqdm(dataset):  # 每篇文章
                src = torch.tensor(self.tokenizer.encode(document)).unsqueeze(dim=0)
                batch = self.batch(src=src, trg=src)
                src_mask = batch.src_mask
                tgt_mask = batch.trg_mask
                data.append(src)
                document_src_mask.append(src_mask)
                document_tgt_mask.append(tgt_mask)
                pass
            document_dataset.append(data)
            document_src.append(document_src_mask)
            document_tgt.append(document_tgt_mask)
            self.size.append(len(data))  # 记录数据集大小
        return document_dataset, document_src, document_tgt

    def read_image(self):
        images = self.loaddata.load_image()  # 加载图片
        image_data = []
        print('read_image: now')
        for order in range(len(self.size)):
            document_image = []
            image_size = len(images[order])
            random = torch.randperm(len(images[order]), dtype=torch.int, device='cpu')  # 随机抽样
            for number in tqdm(range(self.size[order])):
                sample_data = []
                for sample in range(number, number + self.image_number):
                    sample_data.append(images[order][random[sample % image_size]])  # 抽样图片
                document_image.append(
                    torch.cat(sample_data, dim=0).view(self.image_number, 3, 224, 224))
                pass
            image_data.append(document_image)
        return image_data

    def read_trees(self):
        parent = self.loaddata.load_dependencytree()  # 加载parent语句
        trees = []
        print('read_trees: now')
        for dataset in parent:  # 数据集
            documents = []
            for data in tqdm(dataset):  # 每篇文章
                document = []
                for node in data:
                    document.append(self.read_tree(node))
                documents.append(document)
                pass
            trees.append(documents)
        return trees

    def read_tree(self, line):
        parents = line
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = CreateTree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self):
        labels = deepcopy(self.documents)
        return labels


"""
from dataprocess.loaddata import Load_data
from pytorch_transformers import BertTokenizer

token = BertTokenizer.from_pretrained('D:\Homework\pretrained_model\BERT_base_uncased\Bert-base-uncased')

train = MutimodelDataset(Load_data(), token)

sentences, parents = train[0]

jishu = 0
for i in tqdm(range(len(sentences))):
    for j in range(len(sentences[i])):
        sentence = sentences[i][j]
        parent = parents[i][j]
        if (len(parent) > len(sentence)):
            print(sentence, parent)
            jishu += 1
print(jishu)
"""
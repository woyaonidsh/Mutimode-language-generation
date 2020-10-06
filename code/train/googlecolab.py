import os
import json
import skimage
import skimage.transform
import skimage.io
import gc
from PIL import Image
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable
from pytorch_transformers import GPT2Tokenizer

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

class Bottleneck(nn.Module):
    # 第二种连接方式，这种参数少，适合层数超多的 resnet50及以上
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # 所有网络都是通过这个类产生的，只要传入不同的参数即可
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 训练数据的embedding
        self.tgt_embed = tgt_embed  # 目标数据的embedding
        self.generator = generator  # 预测每个词的概率

    #        self.linner = nn.Linear(512, 512)

    def forward(self, src, tgt, src_mask, tgt_mask, image):
        # "Take in and process masked src and target sequences."
        textencoder = self.encode(src, src_mask)
        final = textencoder + image  # 进行线性变化
        return self.generator(self.decode(final, src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# def textandimage(text , image):    # TODO 文本信息与图片信息进行融合
#    linner = nn.Linear(512 , 512)
#    result = linner(text+image)
#    return result


class Generator(nn.Module):  # 生成文本
    # "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.softmax(self.proj(x), dim=-1)


"""
        text = []
        for i in result:   # 6个batch
            for j in i:
                text.append(torch.argmax(j).item())
        result = torch.tensor(text).view(6,512)     # 得到文本的序列号
        return result
"""


def clones(module, N):  # The encoder is composed of a stack of  N=6 identical layers.
    # "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):  # Transformer的encoder层
    # "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    # "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    # "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    # "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    # "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    # "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    # "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# 实现Transformer模型
def make_model(src_vocab=10, tgt_vocab=10, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(  # 对encoder-decoder的每部分进行初始化
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    # "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :]
            self.trg_y = trg[:, :]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    #    @staticmethod
    def make_std_mask(self, tgt, pad):
        # "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


data = processdata()  # 数据预处理函数
valtext_data = data[0]  # 得到验证集的文本+图片
tokenizer = GPT2Tokenizer.from_pretrained('D:\homework\pretrained_model\GPT2')  # 使用GPT2的编码器
# 'D:\homework\pretrained_model\GPT2')  # 使用GPT2的编码器

del data  # 清内存
gc.collect()  # 回收内存

valtext_token = []  # 验证集文本GPT2的编码
valimage_token = []  # 验证集图片的tensor

for textdata in valtext_data:
    str_text = ''
    for i in textdata[0:-3]:  # 拼接文本
        str_text += i
    valtext_token.append(tokenizer.encode(str_text))  # 使用GPT2进行编码
    valimage_token.append(textdata[-3:])
print('The data process is finish')


def paddingtensor(valtext):  # 对token进行padding,补足为3072个，不够的用50256终止符进行补充
    val_tensor = []
    for i in valtext:
        c_i = len(i)
        for j in range(3072 - c_i):
            i.append(50256)
        val_tensor.append(i)
    #        val_tensor.append(torch.tensor(i).view(6, 512))
    return val_tensor


valtext_token = paddingtensor(valtext_token)  # 得到padding后的文本


# valimage_tensor = torch.tensor(valimage_token).cuda()  # 得到图片的tensor

# print(valtext_tensor.shape)
# print(valimage_tensor.shape)

class mutimodel(nn.Module):  # 多模态的模型
    def __init__(self, resnet50, Transformer):
        super(mutimodel, self).__init__()
        self.resnet = resnet50
        self.transformer = Transformer
        self.lineal = nn.Linear(1000, 262144)

    def forward(self, src, tgt, src_mask, tgt_mask, image):
        imagetensor = self.resnet(image)
        imagetensor = self.lineal(imagetensor).view(6, 512, 512)  # 对图片进行维度转置
        return self.transformer(src, tgt, src_mask, tgt_mask, imagetensor)  # 得到最终的结果


resnet = resnet50().cuda()
transformer = make_model(50257,50257,2).cuda()

model = mutimodel(resnet50(), make_model(50257, 50257))  # 得到模型

Loss = torch.nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 定义优化器

gc.collect()  # 回收内存

print('The model will start training')

# TODO 开始训练
train_length = len(valtext_token)  # 数据总数
epoch = 1  # 训练轮数
for ii in range(epoch):
    total_loss = 0
    jishu = 0
    for k in range(train_length):
        textbatch, imagebatch = valtext_token[k], valimage_token[k]
        src = torch.tensor(textbatch).view(6, 512)  # 得到文本向量
        imagetensor = torch.tensor(imagebatch, dtype=torch.float).view(3, 3, 224, 224)  # 得到图片的向量
        batch = Batch(src=src, trg=src)
        src_mask = batch.src_mask
        tgt_mask = batch.trg_mask
        output = model(src, src, src_mask, tgt_mask, torch.cat([imagetensor, imagetensor]))  # 模型的输出
        loss = Loss(output.view(6 * 512, 50257), src.view(6 * 512))  # 计算loss
        total_loss += loss
        loss.backward()  # 反向求导
        optimizer.step()  # 更新参数
        print(loss)
        jishu += 1
        if jishu % 2000 == 0:
            print('Loss is : '.format(.2), total_loss / jishu)
print('The model training is end')


def wirte_text(text):  # 续写文本
    total_text = text
    index_token = tokenizer.encode(text)
    tokens_tensor = torch.tensor([index_token]).cuda()
    for _ in range(500):
        with torch.no_grad():
            output_index = model(tokens_tensor)
        predicted_index = torch.argmax(output_index[0, -1, :]).item()
        total_text += tokenizer.decode([predicted_index])
        if '<|endoftext|>' in total_text:
            break
        index_token += [predicted_index]
        tokens_tensor = torch.tensor([index_token]).cuda()
    print(total_text)  # 生成最终文本

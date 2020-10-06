import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset  # 用来加载数据
from pytorch_transformers import GPT2Tokenizer
import json
from imagemodel import resnet50  # 加载resnet50模型
from imagemodel import Transformer  # 加载Transformer模型
from dataprocess import datapeocess  # 数据预处理
import gc

tokenizer = GPT2Tokenizer.from_pretrained('D:\homework\pretrained_model\GPT2')  # 使用GPT2的编码器

data = datapeocess.processdata()  # 数据预处理函数
valtext_data = data[0]  # 得到验证集的文本+图片
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
        self.lineal1 = nn.Linear(1000, 128)
        self.lineal2 = nn.Linear(3 * 2, 24 * 2)
        self.lineal3 = nn.Linear(1, 128)

    def forward(self, src, tgt, src_mask, tgt_mask, image, training):
        if (training == True):  # 判断是否训练
            imagetensor = self.lineal1(self.resnet(image))  # 对图片进行维度扩充
            imagetensor = self.lineal2(imagetensor.transpose(1, 0)).transpose(1, 0).unsqueeze(dim=2)
            imagetensor = self.lineal3(imagetensor)  # 进行维度变化
            return self.transformer(src, tgt, src_mask, tgt_mask, imagetensor)  # 得到最终的结果
        else:
            return self.transformer(src, tgt, src_mask, tgt_mask, image)  # 得到最终的结果


# resnet = resnet50.resnet50().cuda()
# transformer = Transformer.make_model().cuda(50257, 50257,1).cuda()

model = mutimodel(resnet50.resnet50(), Transformer.make_model(50257, 50257, 3, d_model=128, d_ff=1024))  # 得到mini模型
model.cuda()

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
    for k in range(0, train_length, 2):
        textbatch, imagebatch = valtext_token[k:k + 2], valimage_token[k:k + 2]
        src = torch.tensor(textbatch).view(24 * 2, 128).cuda()  # 得到文本向量
        i_tensor = torch.tensor(imagebatch, dtype=torch.float).view(3 * 2, 3, 224, 224).cuda()  # 得到图片的向量
        batch = Transformer.Batch(src=src, trg=src)
        src_mask = batch.src_mask.cuda()
        tgt_mask = batch.trg_mask.cuda()
        #        two_image = torch.cat([i_tensor, i_tensor,i_tensor,i_tensor,i_tensor,i_tensor,i_tensor,i_tensor]).cuda()
        output = model(src, src, src_mask, tgt_mask, i_tensor, True)  # 模型的输出
        loss = Loss(output.view(6 * 512 * 2, 50257), src.view(6 * 512 * 2))  # 计算loss
        total_loss += loss
        loss.backward()  # 反向求导
        optimizer.step()  # 更新参数
        print(jishu)
        jishu += 1
        if jishu % 2000 == 0:
            print('Loss is : '.format(.2), total_loss / jishu)
print('The model training is end')

mytext = 'China is a very beautiful country, with vast land and abundant resources, beautiful mountains and clear waters,'


def wirte_text(text):  # 续写文本
    total_text = text
    index_token = tokenizer.encode(text)
    tokens_tensor = torch.tensor([index_token]).cuda()
    image_tensor = torch.randn(1, 20, 128).cuda()
    for _ in range(200):
        with torch.no_grad():
            batch = Transformer.Batch(src=tokens_tensor, trg=tokens_tensor)
            tokens_mask = batch.src_mask.cuda()
            tokens_mask2 = batch.trg_mask.cuda()
            output_index = model(tokens_tensor, tokens_tensor, tokens_mask, tokens_mask2, image_tensor, False)
        predicted_index = []
        for i in output_index[0]:
            predicted_index.append(torch.argmax(i[:]).item())
        print(predicted_index)
        total_text += tokenizer.decode(predicted_index)
        if '<|endoftext|>' in total_text:
            break
        #        index_token += [predicted_index]
        tokens_tensor = torch.tensor([predicted_index]).cuda()
    print(total_text)  # 生成最终文本


wirte_text(text=mytext)

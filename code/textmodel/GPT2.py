import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch import nn
import time

tokenizer = GPT2Tokenizer.from_pretrained('D:\homework\pretrained_model\GPT2')
def open_text(file):
    f = open(file , 'r' , encoding='utf-8')
    wenben = f.read()
    f.close()
    return wenben
text = open_text('wenben.txt')
def process_text(text):
    text = text.replace('<s>','')
    text = text.replace('</s>','')
    text = text.replace('\n','')
    return text
text = process_text(text)
index_data = tokenizer.encode(text)
print(len(index_data))
del(text)
dataset_cut = []
for i in range(len(index_data)//256):
    # 将字符串分段成长度为 512
    dataset_cut.append(index_data[i*256:i*256+256])
data_tensor = torch.tensor(dataset_cut).cuda()

#加载训练数据
train_set = TensorDataset(data_tensor,data_tensor)
train_loader = DataLoader(dataset=train_set,batch_size=2,shuffle=False)

model = GPT2LMHeadModel.from_pretrained('D:\homework\pretrained_model\GPT2')   # 加载GPT2模型
model.cuda()
model.train()
pre = time.time()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 定义优化器
jishu = 0
#开始训练
epoch = 30  # 循环学习 30 次
for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx , len(train_loader))
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        loss, logits, _ = model(data, labels=target)
        total_loss += loss
        loss.backward()
        optimizer.step()
        if (batch_idx % 10 == 0):
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/10)
            total_loss = 0
print('训练时间：', time.time()-pre)

# 从下载好的文件夹中加载tokenizer
# 这里你需要改为自己的实际文件夹路径
def write_text(text):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    total = text
    # 预测所有token
    for _ in range(500):
        with torch.no_grad():
            # 将输入tensor输入，就得到了模型的输出，非常简单
            # outputs是一个元组，所有huggingface/transformers模型的输出都是元组
            # 本初的元组有两个，第一个是预测得分（没经过softmax之前的，也叫作logits），
            # 第二个是past，里面的attention计算的key value值
            # 此时我们需要的是第一个值
            outputs = model(tokens_tensor)
            # predictions shape为 torch.Size([1, 11, 50257])，
            # 也就是11个词每个词的预测得分（没经过softmax之前的）
            # 也叫做logits
            predictions = outputs[0]
        # 我们需要预测下一个单词，所以是使用predictions第一个batch，最后一个词的logits去计算
        # predicted_index = 582，通过计算最大得分的索引得到的
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        # 反向解码为我们需要的文本
#        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        total += tokenizer.decode([predicted_index])
        if '<|endoftext|>' in total:
            break
        indexed_tokens += [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        # 解码后的文本：'Who was Jim Henson? Jim Henson was a man'
        # 成功预测出单词 'man'
    print(total)
write_text('I love china, there are a lot of beautiful places,')
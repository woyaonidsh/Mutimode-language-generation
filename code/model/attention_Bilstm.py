import torch
import torch.nn as nn
import random
import math

seed_num = 255  # 随机数种子    TODO

torch.manual_seed(seed_num)  # 设置固定生成随机数的种子，使得每次运行该.py文件时生成的随机数相同
random.seed(seed_num)


class BiLSTM(nn.Module):

    def __init__(self, lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers=2,
                 dropout=0.1, bias=False, negative_slope=0.01):
        super(BiLSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.embed_num = vocabsize  # 词典最长长度
        self.embed_dim = embed_dim
        self.bias = bias  # 为线性层添加偏置
        self.negative_slope = negative_slope  # LeakyRelu的斜率控制
        self.batchsize = batchsize
        # pretrained  embedding
        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, num_layers=self.num_layers, dropout=dropout,
                              bidirectional=True, bias=False, batch_first=True)
        self.layerlist = nn.ModuleList()  # 存放MLP

        for i in range(self.num_layers):  # 添加num_layer层网络
            self.layerlist.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))

        self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

    def forward(self, x):
        bilstm_out, (h, c) = self.bilstm(x)  # 得到bilstm的输出
        h = torch.transpose(h, 1, 0).reshape(-1, self.num_layers, self.hidden_dim)
        h = self.mutilayer_FFN(h)  # 得到经过全连接后的h
        bilstm_out = self.self_attention(bilstm_out)
        output = self.attention(h, bilstm_out)
        return output  # dimension:[batchsize * num_layers * hidden_dim]

    def attention(self, h, all_h):
        d_k = h.size(-1)
        attention_scores = torch.matmul(h, all_h.transpose(-1, -2)) / math.sqrt(d_k)  # 得到注意力矩阵
        p_atten = torch.softmax(attention_scores, dim=-1)  # 按行进行归一化
        output = torch.matmul(p_atten, all_h)
        output = self.leakyrelu(self.layer_2(output))
        return output + h  # 得到最终的h表示

    def self_attention(self, all_h):
        d_k = all_h.size(-1)
        attention_scores = torch.matmul(all_h, all_h.transpose(-1, -2)) / math.sqrt(d_k)
        p_atten = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(p_atten, all_h)
        output = self.leakyrelu(self.layer_1(output))
        return output + all_h  # 得到自注意力后的h

    def mutilayer_FFN(self, h):
        for layer in self.layerlist:
            h = layer(h)
            h = self.leakyrelu(h)
        return h


"""
model = BiLSTM(lstm_hidden_dim=300, vocabsize=200, embed_dim=300, lstm_num_layers=2, dropout=0, batchsize=20).cuda()
data = torch.randn(1,20,300).cuda()

jishu = 0
for i in range(10000):
    output = model(data)
    jishu += 1
    print(jishu)
"""
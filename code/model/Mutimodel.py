import torch
import torch.nn as nn


class Mutimodel(nn.Module):  # 多模态的模型
    def __init__(self, encoder, decoder, generation):
        super(Mutimodel, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.generation = generation  # 文本生成器

    def forward(self, text, text_mask, sentence, tree, image, tgt, tgt_mask):
        x = self.encode(text, text_mask, sentence, tree, image)
        x = self.decode(tgt, x, None, tgt_mask)
        return self.generation(x)

    def encode(self, text, text_mask, sentence, tree, image):
        return self.encoder(text, text_mask, sentence, tree, image)

    def decode(self, tgt, x, src_mask=None, tgt_mask=None):
        return self.decoder(tgt, x, src_mask, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, image_encoder, tran_encoder, treelstm, Bilstm,
                 res_embed, trans_embed, tree_embed, bilstm_embed, negative_slope, device):
        super(Encoder, self).__init__()
        self.image_encoder = image_encoder
        self.tran_encoder = tran_encoder
        self.treelstm = treelstm
        self.Bilstm = Bilstm

        self.device = device

        self.negative_slope = negative_slope

        self.res_embed = res_embed  # 图片的维度
        self.trans_embed = trans_embed  # transformer的encoder维数
        self.tree_embed = tree_embed  # treelstm的维数
        self.bilstm_embed = bilstm_embed  # 输入给bilstm的维度

        self.res_layer = nn.Linear(self.res_embed, self.bilstm_embed)
        self.trans_layer = nn.Linear(self.trans_embed, self.bilstm_embed)
        self.tree_layer = nn.Linear(self.tree_embed, self.bilstm_embed)
        self.active = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

    def forward(self, text, text_mask, sentence, tree, image):
        image_feature = self.image_encoder(image)  # 得到图片特征
        image_feature = self.res_layer(image_feature)

        text_feature = self.tran_encoder(text, text_mask)  # 得到文本特征
        text_feature = self.trans_layer(text_feature)

        sentence_feature = []
        for length in range(len(sentence)):
            sentence_feature.append(self.treelstm(tree[length], sentence[length].to(self.device)))
        sentence_feature = self.tree_layer(torch.cat(sentence_feature, dim=0))

        all_feature = torch.cat([image_feature.unsqueeze(0), text_feature, sentence_feature.unsqueeze(0)],
                                dim=1)  # 拼接所有特征
        return self.Bilstm(all_feature)


class Decoder(nn.Module):
    def __init__(self, tran_decoder, encoder_embed, decoder_embed, negative_slope=0.1):
        super(Decoder, self).__init__()
        self.tran_decoder = tran_decoder
        self.enocder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.negetive_slope = negative_slope

        self.layer = nn.Linear(self.enocder_embed, self.decoder_embed)
        self.active = nn.LeakyReLU(negative_slope=self.negetive_slope)

    def forward(self, tgt, x, src_mask, tgt_mask):
        x = self.active(self.layer(x))
        out = self.tran_decoder(x, src_mask, tgt, tgt_mask)
        return out


class Generator(nn.Module):  # 生成文本
    # "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.softmax(self.project(x), dim=-1).squeeze()  # 压缩维度

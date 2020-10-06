import torch
import torch.nn as nn
from torch.autograd import Variable

class mutimodel(nn.Module):    # 多模态的模型
    def __init__(self , resnet50 , Transformer ):
        super(mutimodel, self).__init__()
        self.resnet = resnet50
        self.transformer = Transformer

    def forward(self , src, tgt , src_mask , tgt_mask , image):
        return self.transformer(src , tgt , src_mask , tgt_mask , self.resnet(image))   #得到最终的结果

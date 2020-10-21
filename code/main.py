import torch
import torch.nn as nn
from copy import deepcopy
import logging
import json
import os

from dataprocess import unzip
from dataprocess import preprocess
from dataprocess import loaddata
from dataprocess import dataset
from pytorch_transformers import BertTokenizer

from model import TreeLstm
from model import attention_Bilstm
from model import resnet
from model import Transformer
from model import Mutimodel

import train

import config

argus = config.parse_args()


def main(argus):
    argus.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if argus.cuda else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print(device)  # 输出当前设备名
    tokenizer = BertTokenizer.from_pretrained(argus.bert_token_path)  # 加载Bert的token
    # 解压文件
    unzip.unzip_text()
    unzip.unzip_annotation()
    unzip.unzip_image()

    # 预处理文件,如果不需要则跳过,此步时间较长
    preprocessdata = preprocess.preprocess_data(argus.text_path, argus.annotation_path, argus.image_path,
                                                argus.mini_character, argus.spacy_model)
    preprocessdata.get_sentence()
    #    preprocessdata.get_imageannotations()
    preprocessdata.resize_image()

    # 加载数据集
    Dataset = dataset.MutimodelDataset(
        loaddata=loaddata.Load_data(data_path_text=argus.text_path, data_path_annotation=argus.annotation_path,
                                    data_path_image=argus.image_path, datasize=argus.data_size),
        tokenizer=tokenizer, image_number=argus.image_number, make_batch=Transformer.Batch)

    # construct neural network
    treelstm = TreeLstm.SimilarityTreeLSTM(vocab_size=argus.vocab_size, in_dim=argus.tree_embed_dim,
                                           mem_dim=argus.tree_mem_dim,
                                           hidden_dim=argus.tree_hidden_dim, sparsity=argus.sparse,
                                           freeze=argus.freeze_embed)
    # tree-lstm
    resnet50 = resnet.resnet50(save_path=argus.save_resnet, pretrained=argus.pretrain_resnet,
                               image_embed=argus.image_embed)
    # resnet
    attention_bilstm = attention_Bilstm.BiLSTM(lstm_hidden_dim=argus.lstm_hidden_dim, vocabsize=argus.vocab_size,
                                               embed_dim=argus.lstm_embed_dim, lstm_num_layers=argus.lstm_num_layers,
                                               batchsize=argus.batch_size, negative_slope=argus.LeakyRelu_slope)
    tran_en_ffn = Transformer.PositionwiseFeedForward(d_model=argus.d_model, d_ff=argus.d_ff, dropout=argus.dropout)
    tran_en_mutihead = Transformer.MultiHeadedAttention(h=argus.head, d_model=argus.d_model)
    trans_encodelayer = Transformer.EncoderLayer(d_model=argus.d_model, self_attn=tran_en_mutihead,
                                                 feed_forward=tran_en_ffn, dropout=argus.dropout)
    trans_encode = Transformer.Encoder(layer=trans_encodelayer, N=argus.encoder_layer)
    tran_en_embeding = Transformer.Embeddings(d_model=argus.d_model, vocab=argus.vocab_size)
    tran_en_position = Transformer.PositionalEncoding(d_model=argus.d_model, dropout=argus.dropout)
    tran_en_embed = nn.Sequential(tran_en_embeding, tran_en_position)

    trans_encoder = Transformer.Transformer_encoder(encoder=trans_encode, src_embed=tran_en_embed)  # transformer的编码器

    tran_de_ffn = deepcopy(tran_en_ffn)
    tran_de_mutihead1 = deepcopy(tran_en_mutihead)
    tran_de_mutihead2 = deepcopy(tran_en_mutihead)
    tran_decoderlayer = Transformer.DecoderLayer(d_model=argus.d_model, self_attn=tran_de_mutihead1,
                                                 src_attn=tran_de_mutihead2,
                                                 feed_forward=tran_de_ffn, dropout=argus.dropout)
    tran_de_embedding = deepcopy(tran_en_embeding)
    tran_de_position = deepcopy(tran_en_position)
    tran_de_embed = nn.Sequential(tran_de_embedding, tran_de_position)
    trans_decode = Transformer.Decoder(layer=tran_decoderlayer, N=argus.decoder_layer)

    trans_decoder = Transformer.Transformer_decoder(decoder=trans_decode, tgt_embed=tran_de_embed)  # transformer解码器

    # model encoder
    encoder = Mutimodel.Encoder(image_encoder=resnet50, tran_encoder=trans_encoder, treelstm=treelstm,
                                Bilstm=attention_bilstm,
                                res_embed=argus.image_embed, trans_embed=argus.d_model,
                                tree_embed=argus.tree_hidden_dim,
                                bilstm_embed=argus.lstm_embed_dim, negative_slope=argus.LeakyRelu_slope,
                                device=device).to(device)

    # model decoder
    decoder = Mutimodel.Decoder(tran_decoder=trans_decoder, encoder_embed=argus.lstm_hidden_dim,
                                decoder_embed=argus.d_model, negative_slope=argus.LeakyRelu_slope).to(device)

    # model generator
    generator = Mutimodel.Generator(d_model=argus.d_model, vocab=argus.vocab_size).to(device)

    model = Mutimodel.Mutimodel(encoder=encoder, decoder=decoder, generation=generator)

    criterion = nn.CrossEntropyLoss().to(device)  # Loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=argus.lr, weight_decay=argus.wd)  # optimize

    # 构建训练器
    trainer = train.Trainer(model=model, criterion=criterion, optimizer=optimizer, datasize=Dataset.size,
                            batchsize=argus.batch_size, device=device, tokenizer=tokenizer)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Start training
    for epoch in range(argus.epochs):
        train_loss = trainer.train(Dataset[0])
        logger.info('==> Epoch {}, Train \tLoss: {}'.format(epoch, train_loss))

        test_loss, R_score1, R_score2 = trainer.test(Dataset[1])

        checkpoint = {
            'epoch': epoch,
            'Loss ': train_loss,
            'R_1': R_score1,
            'R_2': R_score2,
            'text': test_loss
        }
        logger.debug('==> New optimum found, checkpointing everything now...')
        file = open(os.path.join(argus.save_log, argus.log_file), 'a', encoding='utf-8')
        save = json.dumps(checkpoint)
        file.write(save)
        file.write('\n')
        file.close()


#        torch.save(model.state_dict(), os.path.join(argus.save_log, argus.model_file))


if __name__ == "__main__":
    main(argus=argus)

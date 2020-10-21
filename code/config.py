import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # TODO ---------------dataset------------------  datasets arguments
    parser.add_argument('--text_path', type=str, default='../data/text/', help='path to text')
    parser.add_argument('--image_path', type=str, default='../data/image/', help='path to image')
    parser.add_argument('--annotation_path', type=str, default='../data/image/annotations/',
                        help='path to annotation')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_md', help='spacy language model')
    parser.add_argument('--image_number', type=int, default=3, help='The number of image per text')
    parser.add_argument('--mini_character', type=int, default=0, help='Minimum number of characters in spacy sentence')
    parser.add_argument('--data_size', type=int, default=10, help='The number of data')
    # TODO ---------------Bilstm----------------------
    parser.add_argument('--lstm_hidden_dim', type=int, default=100, help='The dimension of Bilstm')
    parser.add_argument('--lstm_embed_dim', type=int, default=100, help='The dimension of Bilstm embedding')
    parser.add_argument('--lstm_num_layers', type=int, default=4, help='The layer of Bilstm')
    # TODO ---------------resnet-------------------
    parser.add_argument('--save_resnet', type=str, default='pretrained/', help='')
    parser.add_argument('--image_embed', type=int, default=100, help='')  # 选择数据集大小
    parser.add_argument('--pretrain_resnet', action='store_true', help='')
    # TODO ---------------TreeLstm--------------------
    parser.add_argument('--tree_embed_dim', type=int, default=100, help='')
    parser.add_argument('--tree_mem_dim', type=int, default=100, help='')
    parser.add_argument('--tree_hidden_dim', type=int, default=100, help='')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--freeze_embed', action='store_false',
                        help='Freeze word embeddings')
    # TODO ---------------Transformer-----------------
    parser.add_argument('--d_model', type=int, default=128, help='')
    parser.add_argument('--encoder_layer', type=int, default=1, help='')
    parser.add_argument('--decoder_layer', type=int, default=1, help='')
    parser.add_argument('--d_ff', type=int, default=512, help='')
    parser.add_argument('--head', type=int, default=8, help='')
    # TODO ---------------Mutimodel-------------------
    parser.add_argument('--vocab_size', type=int, default=30000, help='The size of vocabulary')
    parser.add_argument('--pre_wordEmbedding', action='store_false', help='')
    parser.add_argument('--paddingId', type=int, default=50257, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--bias', action='store_false', help='')
    parser.add_argument('--LeakyRelu_slope', type=float, default=0.01, help='')
    parser.add_argument('--bert_token_path', type=str, default='pretrained/Bert-base-uncased', help='')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='batchsize for optimizer updates')
    # training arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--save_log', type=str, default='checkpoints/', help='')   # 保存训练信息的文件夹
    parser.add_argument('--log_file', type=str, default='train.txt', help='')  # 保存训练信息的文件名
    parser.add_argument('--model_file', type=str, default='model.pth', help='')  # 保存模型文件名
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', type=bool, default=True, help='')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args

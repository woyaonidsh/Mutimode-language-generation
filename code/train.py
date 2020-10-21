from tqdm import tqdm
import torch
from model import metrics


class Trainer:
    def __init__(self, model, criterion, optimizer, datasize, batchsize, device, tokenizer):
        super(Trainer, self).__init__()
        self.batchsize = batchsize  # 梯度累计
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.datasize = datasize  # 数据集大小
        self.device = device
        self.tokenizer = tokenizer

    # helper function for training
    def train(self, dataset):
        documents, src_masks, sentences, images, trees, targets, tgt_masks = dataset
        self.optimizer.zero_grad()
        total_loss = 0.0
#        indices = torch.randperm(self.datasize[0], dtype=torch.int, device='cpu')  # 返回0-len(dataset)之间数的一个数组
        for id in tqdm(range(self.datasize[0]), desc='Training epoch ' + str(self.epoch + 1) + ''):
#            sample = indices[id]
            document, src_mask, sentence, image, tree, target, tgt_mask = documents[id], src_masks[id], \
                                                                          sentences[id], images[id], trees[
                                                                              id], targets[id], tgt_masks[
                                                                              id]
            document, src_mask, image, target, tgt_mask = document.to(self.device), src_mask.to(
                self.device), image.to(self.device), target.to(self.device), tgt_mask.to(self.device)
            output = self.model(document, src_mask, sentence, tree,
                                image, target, tgt_mask)  # 模型的输入
            loss = self.criterion(output.to(self.device), target.squeeze().to(self.device))
            total_loss += loss.item()
            loss.backward()
            if id % self.batchsize == 0 and id > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                print('Loss is: %.2f' % (total_loss / id))
        self.epoch += 1
        return total_loss / self.datasize[0]

    # helper function for testing
    def test(self, dataset):
        with torch.no_grad():
            documents, src_masks, sentences, images, trees, targets, tgt_masks = dataset
            total_loss = 0.0
            R_1 = 0
            R_2 = 0
            indices = torch.randperm(self.datasize[1], dtype=torch.int, device='cpu')  # 返回0-len(dataset)之间数的一个数组
            all_text = []   # 全部预测的文本
            for id in tqdm(range(self.datasize[1]), desc='Testing epoch  ' + str(self.epoch) + ''):
                sample = indices[id]
                document, src_mask, sentence, image, tree, target, tgt_mask = documents[sample], src_masks[sample], \
                                                                              sentences[sample], images[sample], trees[
                                                                                  sample], targets[sample], tgt_masks[
                                                                                  sample]
                raw_text = document[0:len(document[0])//2]
                raw_sentence = sentence[0:len(sentence)//2]
                raw_tree = tree[0:len(tree)//2]
                raw_text, src_mask, image, target, tgt_mask = raw_text.to(self.device), src_mask.to(
                    self.device), image.to(self.device), target.to(self.device), tgt_mask.to(self.device)
                raw_target = torch.tensor([[0]]).cuda()
                prediction = []
                for length in tqdm(range(len(target[0])//2)):
                    output = self.model(raw_text, None, raw_sentence, raw_tree,
                                        image, raw_target, None)
                    raw_target = torch.argmax(output, dim=-1).unsqueeze(dim=0).unsqueeze(dim=0)
                    prediction.append(raw_target)
                    raw_text = torch.cat([raw_text, raw_target], dim=-1)
                    pass
                predict = torch.cat(prediction, dim=-1).squeeze().tolist()
                text = self.tokenizer.decode(predict)
                all_text.append(text)
                right_text = self.tokenizer.decode(target[0][len(target[0])//2:-1].squeeze().tolist())
                R_score1, R_score2 = metrics.Rouge(model=text, reference=right_text)
                R_1 += R_score1
                R_2 += R_score2
            return all_text, R_1 / self.datasize[1], R_2 / self.datasize[1]
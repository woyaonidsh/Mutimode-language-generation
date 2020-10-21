import torch
import torch.nn as nn


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)

    def forward(self, hidden, state):
        mult_dist = torch.mul(hidden, state)
        abs_dist = torch.abs(torch.add(hidden, -state))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = torch.sigmoid(self.wh(vec_dist))
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim)

    def forward(self, tree, inputs):
        inputs = self.emb(inputs)
        state, hidden = self.childsumtreelstm(tree, inputs)  # 计算左树的特征
        output = self.similarity(hidden, state)
        return output
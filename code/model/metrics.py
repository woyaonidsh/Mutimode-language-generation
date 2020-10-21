import jieba


# 使用jieba进行分词
def Rouge_1(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def Rouge_2(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    terms_reference = jieba.cut(reference)
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


def Rouge(model, reference):
    return Rouge_1(model, reference), Rouge_2(model, reference)

# Rouge("我的世界是光明的","光明给我的世界以力量")

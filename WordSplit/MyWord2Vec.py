from gensim.models.word2vec import Word2Vec
from MyJieba import *


text = get_text()

model = Word2Vec(size=300)
model.build_vocab(text)
model.train(text,total_examples=model.corpus_count,epochs=model.iter)
model.save('test.model')

# 加载模型
# model = Word2Vec.load('test.model')
# 获取词向量
# print(model['支付宝'])
# 计算一个词的最近似的词
# print(model.most_similar(['支付宝']))
# 计算两词之间的余弦相似度
# print(model.similarity('woman','man'))


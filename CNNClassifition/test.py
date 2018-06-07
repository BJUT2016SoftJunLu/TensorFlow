from tensorflow.contrib import learn
import numpy as np

x_text = ['This is a cat', 'This must be boy', 'This is a a dog']
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# 将文本进行编码
x = np.array(list(vocab_processor.fit_transform(x_text)))
# 获取文本字典
print(len(vocab_processor.vocabulary_))

# sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
# # zip(*)解压
# vocabulary = list(zip(*sorted_vocab))[0]
# print(vocabulary)
# print(x)
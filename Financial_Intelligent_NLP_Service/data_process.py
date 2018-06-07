import re
import os
import numpy as np
from tensorflow.contrib import learn

vocab_path = os.path.abspath(os.path.join("", "data/vocab"))


FILE_PATH_1 = os.path.abspath(os.path.join("","data/atec_nlp_sim_train.csv"))
FILE_PATH_2 = os.path.abspath(os.path.join("","data/atec_nlp_sim_train_add.csv"))

NLP_VOCAB_PATH = os.path.abspath(os.path.join("", "data/NLP_vocab"))

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def process_data(neg_file,pos_file,dev_percent=0.1):
    neg_sentence = []
    pos_sentenc = []

    with open(neg_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            neg_sentence.append(clean_str(line.strip()))

    with open(pos_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            pos_sentenc.append(clean_str(line.strip()))

    neg_label = [[0, 1] for _ in neg_sentence]
    pos_label = [[1, 0] for _ in pos_sentenc]

    sentence = np.concatenate((neg_sentence,pos_sentenc),axis=0)
    max_document_length = max([len(s.split(" ")) for s in sentence])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_data = np.array(list(vocab_processor.fit_transform(sentence)))
    vocab_size = len(vocab_processor.vocabulary_)
    Y_data = np.concatenate((neg_label,pos_label),axis=0)

    # 保存词汇表
    vocab_processor.save(vocab_path)

    example = np.concatenate((X_data, Y_data), axis=1)
    print(max_document_length, vocab_size)

    train_size = len(example) - int(dev_percent*len(example))
    x_train = example[:train_size,:max_document_length]
    y_train = example[:train_size,max_document_length:]
    x_dev = example[train_size:,:max_document_length]
    y_dev = example[train_size:,max_document_length:]
    return x_train,y_train,x_dev,y_dev


def NLP_process_data(dev_percent=0.1):
    sentence = []
    Y_label = []

    with open(FILE_PATH_1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            sentence.append(example_split[1] + " " + example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1, 0])
            else:
                Y_label.append([0, 1])

    with open(FILE_PATH_2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            sentence.append(example_split[1] + " " + example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1, 0])
            else:
                Y_label.append([0, 1])

    Y_label = np.array(Y_label)

    max_document_length = max([len(s) for s in sentence])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_data = np.array(list(vocab_processor.fit_transform(sentence)))
    vocab_size = len(vocab_processor.vocabulary_)

    # 保存词汇表
    vocab_processor.save(NLP_VOCAB_PATH)

    train_size = len(X_data) - int(dev_percent * len(X_data))
    x_train = X_data[:train_size]
    y_train = Y_label[:train_size]
    x_dev = X_data[train_size:]
    y_dev = Y_label[train_size:]

    return x_train, y_train, x_dev, y_dev



def batch_iterator(x_train,y_train,batch_size = 64,epoch = 200):

    data_size = x_train.shape
    example = np.concatenate((x_train,y_train),axis=1)
    batch_nums = int(data_size[0]/batch_size) + 1

    for i in range(epoch):
        np.random.shuffle(example)
        x_data = example[:,:data_size[1]]
        y_data = example[:,data_size[1]:]
        for j in range(batch_nums):
            start = j * batch_size
            end = start + batch_size
            if end > data_size[0]:
                yield x_data[start:],y_data[start:]
            else:
                yield x_data[start:end],y_data[start:end]



def main():

    x_train, y_train, x_dev, y_dev = process_data("./data/rt-polarity.neg","./data/rt-polarity.pos")

    return

if __name__ == '__main__':
    main()
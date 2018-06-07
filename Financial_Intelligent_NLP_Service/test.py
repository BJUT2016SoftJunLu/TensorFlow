from tensorflow.contrib import learn
import os
import numpy as np


FILE_PATH_1 = os.path.abspath(os.path.join("","data/atec_nlp_sim_train.csv"))
FILE_PATH_2 = os.path.abspath(os.path.join("","data/atec_nlp_sim_train_add.csv"))

NLP_VOCAB_PATH = os.path.abspath(os.path.join("", "data/NLP_vocab"))

def NLP_process_data(dev_percent=0.1):
    sentence = []
    Y_label = []

    with open(FILE_PATH_1,'r',encoding='utf-8') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            sentence.append(example_split[1]+" "+example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1,0])
            else:
                Y_label.append([0,1])


    with open(FILE_PATH_2,'r',encoding='utf-8') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            sentence.append(example_split[1]+" "+example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1,0])
            else:
                Y_label.append([0,1])

    Y_label = np.array(Y_label)

    max_document_length = max([len(s) for s in sentence])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_data = np.array(list(vocab_processor.fit_transform(sentence)))
    vocab_size = len(vocab_processor.vocabulary_)
    print(max_document_length,vocab_size)

    # 保存词汇表
    vocab_processor.save(NLP_VOCAB_PATH)

    train_size = len(X_data) - int(dev_percent * len(X_data))
    x_train = X_data[:train_size]
    y_train = Y_label[:train_size]
    x_dev = X_data[train_size:]
    y_dev = Y_label[train_size:]

    return x_train,y_train,x_dev,y_dev


def main():
    x_train, y_train, x_dev, y_dev = NLP_process_data()
    print(len(x_train))
    print(len(y_train))
    print(x_train[0])
    print(y_train)

if __name__ == '__main__':
    main()
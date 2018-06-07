import os
import sys
import tensorflow as tf
from Financial_Intelligent_NLP_Service.data_process import *
import Financial_Intelligent_NLP_Service.model as model
import pickle

vocab_path = os.path.abspath(os.path.join("", "data/NLP_vocab"))
model_path = os.path.abspath(os.path.join("", "model/"))

# def predict(test_path):
#     test_sentence = []
#     test_resutlt = []
#     with open(test_path,'r',encoding='utf-8') as f:
#         for line in f.readlines():
#             test_sentence.append(clean_str(line.strip()))
#
#     # 加载词汇表
#     vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     x_test = np.array(list(vocab_processor.transform(test_sentence)))
#
#     # 加载模型
#     text_cnn = model.TextCNN(max_document_length=56,
#                              vocab_size=18758,
#                              embedding_size=128,
#                              filter_size=[3,4,5],
#                              class_nums=2,
#                              filter_channel=128,
#                              l2_reg_lambda=0.0,
#                              learning_rate=1e-3)
#     session = tf.Session()
#     saver = tf.train.Saver()
#     check_point = tf.train.latest_checkpoint(model_path)
#     saver.restore(session,check_point)
#
#     test_dict = {
#         text_cnn.input_x:x_test,
#         text_cnn.keep_prob:1.0
#     }
#     test_prob = session.run(text_cnn.prob,feed_dict=test_dict)
#     for index in np.argmax(test_prob,1):
#         if index == 0:
#             test_resutlt.append(1)
#         else:
#             test_resutlt.append(0)
#
#     return test_resutlt



def change_pickle_protocol(filepath,protocol=2):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    with open(filepath,'wb') as f:
        pickle.dump(obj,f,protocol=protocol)


def predict(INPUT_PATH,OUTPUT_PATH):
    predict_sentence = []
    line_number = []
    predict_resutlt = []
    with open(INPUT_PATH,'r',encoding='utf-8') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            line_number.append(int(example_split[0]))
            predict_sentence.append(example_split[1] + " " + example_split[2])

    # 加载词汇表
    change_pickle_protocol(vocab_path)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(predict_sentence)))
    print(x_test.shape)

    # 加载模型
    text_cnn = model.TextCNN(max_document_length=167,
                             vocab_size=230611,
                             embedding_size=128,
                             filter_size=[3,4,5],
                             class_nums=2,
                             filter_channel=128,
                             l2_reg_lambda=0.0,
                             learning_rate=1e-3)

    session = tf.Session()
    saver = tf.train.Saver()
    check_point = tf.train.latest_checkpoint(model_path)
    saver.restore(session,check_point)

    test_dict = {
        text_cnn.input_x:x_test,
        text_cnn.keep_prob:1.0
    }
    test_prob = session.run(text_cnn.prob,feed_dict=test_dict)
    for index in np.argmax(test_prob,1):
        if index == 0:
            predict_resutlt.append(1)
        else:
            predict_resutlt.append(0)

    line_number = np.array(line_number)
    predict_resutlt = np.array(predict_resutlt)
    line_number = line_number.reshape((len(line_number),1))
    predict_resutlt = predict_resutlt.reshape((len(predict_resutlt),1),)
    print(line_number)
    print(predict_resutlt)
    predict_resutlt = np.concatenate([line_number,predict_resutlt],axis=1)
    print(predict_resutlt)


    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for line in predict_resutlt:

            f.write(str(line[0]) + "\t" + str(line[1]) +"\n")
    return

def main():
    args = sys.argv
    predict("./data/test_predict","./data/result")
    return

if __name__ == '__main__':
    main()

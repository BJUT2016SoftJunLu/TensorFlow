#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import re


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


# def preprocess():
#
#     neg_sentences = []
#     pos_sentences = []
#     with open("./data/rt-polarity.neg",'r',encoding="utf-8") as f:
#         for line in f.readlines():
#             neg_sentences.append(clean_str(line.strip()))
#
#     with open("./data/rt-polarity.pos",'r',encoding="utf-8") as f:
#         for line in f.readlines():
#             pos_sentences.append(clean_str(line.strip()))
#
#     x_text = neg_sentences + pos_sentences
#     neg_label = [[1,0] for _ in neg_sentences]
#     pos_label = [[0,1] for _ in pos_sentences]
#
#     Y_label = np.concatenate([pos_label, neg_label], 0)
#
#     # Build vocabulary
#     max_document_length = max([len(x.split(" ")) for x in x_text])
#     vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#     x = np.array(list(vocab_processor.fit_transform(x_text)))
#
#     max_sentence_length = max(len(sentence.split(" ")) for sentence in x_text)
#     vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
#     X_data = np.array(list(vocab_processor.fit_transform(x_text)))
#
#     x_train,x_dev =  X_data[:len(X_data) - int(0.1 * len(X_data))], X_data[len(X_data) - int(0.1 * len(X_data)):]
#     y_train,y_dev =  Y_label[:len(X_data) - int(0.1 * len(X_data))],Y_label[len(X_data) - int(0.1 * len(X_data)):]
#
#     return x_train, y_train, vocab_processor, x_dev, y_dev,max_document_length
#
#
#
# def batch_iter(x_train, y_train, batch_size, num_epochs, shuffle=True):
#
#     data = np.concatenate([x_train, y_train], 1)
#     num_batches_per_epoch = int((x_train.shape[0]-1)/batch_size) + 1
#
#     for epoch in range(num_epochs):
#         np.random.shuffle(data)
#         x_data = data[:,:x_train.shape[1]]
#         y_data = data[:,x_train.shape[1]:]
#         for batch_num in range(num_batches_per_epoch):
#             start = batch_num * batch_size
#             end = start + batch_size
#             if end > x_train.shape[0]:
#                 yield x_data[start:],y_data[start:]
#             else:
#                 yield x_data[start:end],y_data[start:end]



def process_data(neg_file,pos_file,dev_percent=.1):
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

    example = np.concatenate((X_data, Y_data), axis=1)

    train_size = len(example) - int(dev_percent*len(example))
    x_train = example[:train_size,:max_document_length]
    y_train = example[:train_size,max_document_length:]
    x_dev = example[train_size:,:max_document_length]
    y_dev = example[train_size:,max_document_length:]

    return x_train, y_train, vocab_processor, x_dev, y_dev, max_document_length


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




def train(x_train, y_train, vocab_processor, x_dev, y_dev,max_document_length):

    embedding_size = 128
    class_nums = 2
    filter_type = [3,4,5]
    filter_channel = 128
    l2_reg_lambda = 0.0
    model_path = "./model/"
    vocab_size = len(vocab_processor.vocabulary_)

    # placeholder layer
    input_x = tf.placeholder(tf.int32,shape=[None,max_document_length])
    input_y = tf.placeholder(tf.float32,shape=[None,class_nums])
    keep_prob = tf.placeholder(tf.float32)


    # embedding layer
    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


    # conver and pool layer
    pool_output = []
    for size in filter_type:
        filter_W = tf.Variable(tf.truncated_normal(shape=[size,embedding_size,1,filter_channel],stddev=0.1))
        conv = tf.nn.conv2d(embedded_chars_expanded,filter=filter_W,strides=[1,1,1,1],padding="VALID")
        print(conv.shape)
        Bais = tf.Variable(tf.constant(0.1,shape=[filter_channel]))
        a = tf.nn.relu(conv + Bais)

        pool = tf.nn.max_pool(a,ksize=[1,max_document_length - size + 1,1,1],strides=[1,1,1,1], padding="VALID")
        print(pool.shape)
        pool_output.append(pool)

    feature_nums = filter_channel * len(filter_type)
    pool_concat = tf.concat(pool_output,3)
    pool_reshape = tf.reshape(pool_concat,shape=[-1,feature_nums])

    # dropout layer
    dropout_output = tf.nn.dropout(pool_reshape,keep_prob=keep_prob)

    # connected layer
    conn_w = tf.Variable(tf.truncated_normal(shape=[feature_nums,class_nums],stddev=0.1))
    conn_b = tf.Variable(tf.constant(0.1,shape=[class_nums]))
    tf.add_to_collection('losses', tf.nn.l2_loss(conn_w))
    tf.add_to_collection('losses', tf.nn.l2_loss(conn_b))

    conn_output = tf.matmul(dropout_output, conn_w) + conn_b


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(conn_output,1),tf.argmax(input_y,1)),"float"))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conn_output,labels=input_y))
    loss = cross_entropy_loss + tf.add_n(tf.get_collection("losses")) * l2_reg_lambda
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)



    # train layer
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        for x_data, y_data in batch_iterator(x_train,y_train,64, 200):
            feed_dict = {
                input_x: x_data,
                input_y: y_data,
                keep_prob: 0.5
            }
            _, train_loss, acc = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
            print("the step is %s, the train loss is %s, the acc is %s" % (step, train_loss, acc))
            if step % 100 == 0:
                print("\nEvaluation:")
                eval_dict = {
                    input_x: x_dev,
                    input_y: y_dev,
                    keep_prob: 1.0
                }
                eval_loss, acc = sess.run([loss, accuracy], feed_dict=eval_dict)
                print("************* the step is %s, the eval loss is %s, the acc is %s *************" % (step, eval_loss, acc))
                print("")
                tf.train.Saver().save(sess, save_path=model_path, global_step=step)
            step += 1


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, max_document_length = process_data("./data/rt-polarity.neg","./data/rt-polarity.pos")
    train(x_train, y_train, vocab_processor, x_dev, y_dev,max_document_length)

if __name__ == '__main__':
    tf.app.run()
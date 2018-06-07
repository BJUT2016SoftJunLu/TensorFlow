import os
import tensorflow as tf
import datetime
from Financial_Intelligent_NLP_Service.data_process import *
import Financial_Intelligent_NLP_Service.model as model



save_path = os.path.abspath(os.path.join("","model"))
print(save_path+"\\NLP")

def train(x_train, y_train, x_dev, y_dev,text_cnn,dev_step):
    max_acc = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x_data,y_data in batch_iterator(x_train,y_train):
            feed_dict = {
                text_cnn.input_x:x_data,
                text_cnn.input_y:y_data,
                text_cnn.keep_prob:0.5
            }
            _,train_loss,train_acc,step = sess.run([text_cnn.optimizer,text_cnn.loss,text_cnn.accuracy,text_cnn.global_step],feed_dict=feed_dict)
            print("the step is %s , the train loss is %s , the train acc is %s" % (step,train_loss,train_acc))

            current_step = tf.train.global_step(sess, text_cnn.global_step)
            if current_step % dev_step == 0:
                dev_dict = {
                    text_cnn.input_x: x_dev,
                    text_cnn.input_y: y_dev,
                    text_cnn.keep_prob: 1.0
                }
                dev_loss, dev_acc = sess.run([text_cnn.loss, text_cnn.accuracy], feed_dict=dev_dict)
                print("")
                print("the current_step is %s , the dev loss is %s , the dev acc is %s  ***********************" % (current_step, dev_loss, dev_acc))
                print("")
                if dev_acc > max_acc:
                    tf.train.Saver().save(sess, save_path=save_path+"\\NLP", global_step=current_step)
                    max_acc = dev_acc

    return



def main():

    # x_train, y_train, x_dev, y_dev = process_data("./data/rt-polarity.neg","./data/rt-polarity.pos")
    x_train, y_train, x_dev, y_dev = NLP_process_data()
    # 56 18758
    # 167 230611
    text_cnn = model.TextCNN(max_document_length=167,
                             vocab_size=230611,
                             embedding_size=128,
                             filter_size=[3,4,5],
                             class_nums=2,
                             filter_channel=128,
                             l2_reg_lambda=0.0,
                             learning_rate=1e-3)

    train(x_train, y_train, x_dev, y_dev,text_cnn,dev_step=100)
    return

if __name__ == '__main__':
    main()
import  tensorflow as tf
import numpy as np
from crnn_data import *

class CRNNModel:
    def __init__(self,lstm_size,lstm_layer,learning_rate,keep_prop,
                 init_state_size,batch_line,time_step,class_nums,
                 data_generator=None,save_every_batch=None,log_every_batch=None,
                 save_model_path=None,load_model_directory=None):
        self.lstm_size = lstm_size
        self.lstm_layer = lstm_layer
        self.learning_rate = learning_rate
        self.keep_prop = keep_prop
        self.init_state_size = init_state_size
        self.batch_line = batch_line
        self.time_step = time_step
        self.class_nums = class_nums
        self.data_generator = data_generator
        self.save_every_batch = save_every_batch
        self.log_every_batch = log_every_batch
        # 指定模型保存路径,要指定模型文件的名称
        self.model_path = save_model_path
        self.load_model_directory = load_model_directory
        return

    def build(self):
        self.input_X = tf.placeholder(np.int32, shape=(self.batch_line, self.time_step))
        self.input_Y = tf.placeholder(np.int32, shape=(self.batch_line, self.time_step))

        def get_cell(lstm_size, keep_prop):
            base_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
            lstm_drop = tf.nn.rnn_cell.DropoutWrapper(base_lstm, keep_prop)
            return lstm_drop

        multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [get_cell(self.lstm_size, self.keep_prop) for _ in range(self.lstm_layer)])
        self.init_state = multi_lstm.zero_state(self.init_state_size, np.float32)

        # 对input_X进行one_hot编码
        input_X_onehot = tf.one_hot(self.input_X, self.class_nums)
        outputs, self.final_state = tf.nn.dynamic_rnn(multi_lstm, input_X_onehot, initial_state=self.init_state)

        output_W = tf.Variable(tf.truncated_normal(shape=[self.lstm_size, self.class_nums], stddev=0.1),dtype=tf.float32)
        output_B = tf.Variable(tf.zeros(shape=[self.class_nums]))
        logits = tf.matmul(tf.reshape(outputs, shape=[self.batch_line * self.time_step, self.lstm_size]),output_W) + output_B
        self.output_prob = tf.nn.softmax(logits)

        # 对input_Y进行one_hot编码
        input_Y_onehot = tf.reshape(tf.one_hot(self.input_Y, self.class_nums), shape=logits.shape)
        # 定义损失函数
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_Y_onehot))
        # 对梯度进行剪切，防止梯度爆炸
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
        return

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.init_state)
            for X,Y in self.data_generator:
                step = step + 1
                feed_dict={
                    self.input_X: X,
                    self.input_Y: Y,
                    self.init_state:new_state
                }
                loss_info,new_state,_= sess.run([self.loss,self.final_state,self.optimizer],feed_dict=feed_dict)
                if step % self.log_every_batch == 0:
                    print("the step is: %s,  the loss is: %s" % (step,loss_info))
                if step % self.save_every_batch == 0:
                    # 保存当前正在训练的模型
                    tf.train.Saver().save(sess,save_path=self.model_path,global_step=step)
            tf.train.Saver().save(sess, save_path=self.model_path, global_step=step)


    def load_model(self):
        self.session = tf.Session()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.load_model_directory)
        saver.restore(self.session,checkpoint)
        return

    def predict(self,init_char_number,predict_length,number_char_dict):
        result_number = []
        new_state = self.session.run(self.init_state)
        prob = np.ones((self.class_nums,))
        input_char = np.zeros((1,1))
        input_char[0,0] = init_char_number
        for i in range(predict_length):
            if i is not 0:
                input_char = np.zeros((1, 1))
                input_char[0, 0] = prob_number
            feed_dict ={
                self.input_X:input_char,
                self.init_state:new_state,
            }
            prob,new_state = self.session.run([self.output_prob,self.final_state],feed_dict=feed_dict)
            prob_number = self.prob_to_number(prob)
            result_number.append(prob_number)
        result_text = []
        for n in result_number:
            if n == len(number_char_dict.keys()):
                result_text.append("<unk>")
            else:
                result_text.append(number_char_dict[n])
        return "".join(result_text)

    def prob_to_number(self,prob):
        prob = np.squeeze(prob)
        prob[np.argsort(prob)[:5]] = 0
        # 归一化，保证概率和为1
        prob = prob/np.sum(prob)
        return np.random.choice(self.class_nums,1,p=prob)[0]
import  tensorflow as tf
from code.data_generation import *

class SCModel:
    def __init__(self,hidden_layer,learn_rate,batch_rows,max_length,class_nums,
                 max_iteration=None,save_model_path=None,load_model_directory=None):
        self.hidden_layer = hidden_layer
        self.leran_rate =learn_rate
        self.batch_rows = batch_rows
        self.class_nums = class_nums
        self.max_length = max_length
        self.max_iteration = max_iteration
        self.save_model_path = save_model_path
        self.load_model_directory = load_model_directory
        self.input_X = tf.placeholder(tf.float32, shape=(self.batch_rows, self.max_length, 1))
        self.input_Y = tf.placeholder(tf.float32, shape=(self.batch_rows, self.class_nums))
        self.real_length = tf.placeholder(tf.int32, shape=(self.batch_rows))
        return

    def build(self):

        base_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_layer)
        # sequence_length参数会判断真实长度，从而提前终止计算 (optional) An int32/int64 vector sized `[batch_size]`
        outputs,final_state = tf.nn.dynamic_rnn(base_cell,self.input_X,sequence_length=self.real_length,dtype=tf.float32)
        # tensorflow中不支持据索引对outputs取值
        # 在tensorflow中，不能直接根据real_length取output  gather的功能类似于根据索引取值
        # tensorflow不会自动进行类型转换
        index= tf.range(0,self.batch_rows) * self.max_length + (self.real_length - 1)
        output = tf.gather(tf.reshape(outputs,shape=[-1,self.hidden_layer]),index)


        W = tf.Variable(tf.truncated_normal(shape=[self.hidden_layer,self.class_nums],stddev=0.1),dtype=tf.float32)
        B = tf.Variable(tf.zeros(shape=[self.class_nums]),dtype=tf.float32)

        logits = tf.matmul(output,W) + B
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_Y,logits=logits))
        optimizer = tf.train.AdamOptimizer(self.leran_rate).minimize(loss)
        prob = tf.nn.softmax(logits)
        return loss,optimizer,prob

    def train(self,loss,optimizer):
        tsd = ToySequenceData()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 1
            min_loss = float('inf')
            life = 3
            while True:
                batch_data, batch_labels, batch_seqlen = tsd.next(self.batch_rows)
                batch_data = np.array(batch_data)
                batch_labels = np.array(batch_labels)
                batch_seqlen = np.array(batch_seqlen)

                feed_dict = {
                    self.input_X:batch_data,
                    self.input_Y:batch_labels,
                    self.real_length:batch_seqlen
                }
                # 注意变量值覆盖
                new_loss,_ = sess.run([loss,optimizer],feed_dict=feed_dict)
                if step % 50 == 0:
                    if life == 0:
                        print(" 150 batchs loss not reduce, stop training....")
                        break
                    if min_loss < new_loss:
                        min_loss = new_loss
                        life -=  1
                    else:
                        life = 3
                    print("the step is %s,  the loss is %s" % (step,new_loss))
                if step % 1000 == 0:
                    tf.train.Saver().save(sess,self.save_model_path,global_step=step)
                step += 1
                if step >= self.max_iteration:
                    break
            tf.train.Saver().save(sess, self.save_model_path, global_step=step)
        return

    def load_model(self):
        session = tf.Session()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.load_model_directory)
        saver.restore(session, checkpoint)
        return session

    def predict(self,session,prob,test_data,test_seqlen):
        feed_dict = {
            self.input_X:test_data,
            self.real_length:test_seqlen,
        }
        result = session.run([prob],feed_dict=feed_dict)
        return result

def main():


    sc_model = SCModel(hidden_layer=64,
                       learn_rate=0.01,
                       batch_rows=500,
                       max_length=20,
                       class_nums=2,
                       max_iteration=10000,
                       save_model_path="../model/sc_model",
                       load_model_directory="../model/")

    # sc_model.train(loss,optimizer)

    tsd = ToySequenceData(n_samples=500, max_seq_len=20)
    test_data = np.array(tsd.data)
    test_seqlen = np.array(tsd.seqlen)

    _, _, prob = sc_model.build()
    session=  sc_model.load_model()
    prob_result = sc_model.predict(session,prob,test_data,test_seqlen)[0]
    class_result = np.zeros_like(prob_result)
    for i in range(prob_result.shape[0]):
        index = np.argmax(prob_result[i])
        class_result[i][index] = 1
    print(class_result)
    return

if __name__ == '__main__':
    main()




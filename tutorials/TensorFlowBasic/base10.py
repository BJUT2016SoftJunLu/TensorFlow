import  tensorflow as tf
import  numpy as np
from tensorflow.examples.tutorials.mnist import input_data

"""
Multi(dynamic or static) base RNN 图像分类
"""

"""
input layer
"""
input_W = tf.Variable(tf.truncated_normal([28,128],stddev=0.1),dtype=np.float32)
input_B = tf.Variable(tf.constant(0.1,shape=[128]),dtype=np.float32)

input_X = tf.placeholder(np.float32,[None,28,28])
input_Y = tf.placeholder(np.float32,[None,10])

input_datas = tf.matmul(tf.reshape(input_X,shape=[100*28,28]),input_W) + input_B



"""
cell layer
"""
base_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 创建Multi Cell
multi_cell = tf.nn.rnn_cell.MultiRNNCell([base_cell]*2)
# 初始化状态
init_state = multi_cell.zero_state(100,dtype=np.float32)

# input_datas = tf.reshape(input_datas,shape=[28,100,128])
# input_datas = tf.unstack(input_datas,axis=0)
# outputs,final_state = tf.nn.static_rnn(base_cell,input_datas,initial_state=init_state)


input_datas = tf.reshape(input_datas,shape=[100,28,128])
# outputs 为最后一列每一个cell的output组成的Tensor对象 final_stat为最后一列每一个cell的state组成的tuple
outputs,final_state = tf.nn.dynamic_rnn(multi_cell,input_datas,initial_state=init_state)
print(final_state[-1].shape)



"""
output layer
"""
output_W = tf.Variable(tf.truncated_normal([128,10],stddev=0.1),dtype=np.float32)
output_B = tf.Variable(tf.constant(0.1,shape=[10]),dtype=np.float32)
output_datas = tf.matmul(final_state[-1],output_W) + output_B
output_types = tf.nn.softmax(output_datas)

"""
loss and optimizer
"""
loss = -tf.reduce_mean(input_Y * tf.log(output_types))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


"""
train and test
"""
init = tf.global_variables_initializer()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        if i % 50 != 0:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = batch_xs.reshape(100, 28, 28)
            sess.run(optimizer, feed_dict={input_Y: batch_ys, input_X: batch_xs})
        else:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = batch_xs.reshape(100, 28, 28)
            print(sess.run(loss, feed_dict={input_Y: batch_ys, input_X: batch_xs}))
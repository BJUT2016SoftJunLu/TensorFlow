import  tensorflow as tf
import  numpy as np
from tensorflow.examples.tutorials.mnist import input_data

"""
dynamic base RNN 图像分类
static  base RNN 图像分类
"""

""""
input layer
"""
input_W = tf.Variable(tf.truncated_normal([28,128],stddev=0.1),dtype=np.float32)
input_B = tf.Variable(tf.constant(0.1,shape=[128],dtype=np.float32))

input_X = tf.placeholder(np.float32,shape=[None,28,28])
input_Y = tf.placeholder(np.float32,shape=[None,10])

input_datas = tf.matmul(tf.reshape(input_X,shape=[100*28,28]),input_W) + input_B

"""
cell layer
"""
base_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
init_state = base_cell.zero_state(100,dtype=np.float32)
# inputs=大小为[batch_size,time_step,input_size]的Tensor对象 if time_major=False
# inputs=大小为[time_step,batch_size,input_size]的Tensor对象 if time_major=True
input_datas = tf.reshape(input_datas,[100,28,128])
outputs,final_state = tf.nn.dynamic_rnn(base_cell,input_datas,initial_state=init_state)


# inputs=大小为time_step的列表,其中元素大小为[batch_size, input_size]的Tensor对象
# outputs,final_state = tf.nn.static_rnn(base_cell,input_datas,initial_state=init_state)


"""
ouput layer
"""
output_W = tf.Variable(tf.truncated_normal([128,10],stddev=0.1),dtype=np.float32)
output_B = tf.Variable(tf.constant(0.1,shape=[10],dtype=np.float32))
output_datas = tf.matmul(final_state,output_W) + output_B
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
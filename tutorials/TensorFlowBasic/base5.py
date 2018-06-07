import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
分类(添加dropout)
"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def get_accuracy(test_x,test_y):
    global output
    test_result = sess.run(output,feed_dict={input_x:test_x,input_y:test_y,keep_prob:1.0})
    # argmax获取最大值索引
    compare_result = tf.equal(tf.argmax(test_result,1) ,tf.argmax(test_y,1))
    accuracy = tf.reduce_mean(tf.cast(compare_result,tf.float32))
    return accuracy

def add_layer(x_data,input_size,out_size,active_function=None):
    weights = tf.Variable(tf.random_normal([input_size,out_size]))
    bais = tf.Variable(tf.constant(0.01,shape=[out_size]))
    z = tf.matmul(x_data,weights) + bais
    # 添加dropout
    z = tf.nn.dropout(z,keep_prob=keep_prob)
    if active_function is None:
        return z
    else:
        return active_function(z)

keep_prob = tf.placeholder(tf.float32)
input_x = tf.placeholder(tf.float32,[None,784])
input_y = tf.placeholder(tf.float32,[None,10])

layer1 = add_layer(input_x,784,50,active_function=tf.nn.tanh)
output = add_layer(layer1,50,10,active_function=tf.nn.softmax)

# 定义交叉熵损失函数
loss = -tf.reduce_mean(input_y * tf.log(output))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={input_x:batch_xs,input_y:batch_ys,keep_prob:0.5})
        if i % 100 == 0:
            print(sess.run(loss,feed_dict={input_x:mnist.test.images,input_y:mnist.test.labels,keep_prob:1.0}))
            print(sess.run(get_accuracy(mnist.test.images, mnist.test.labels)))

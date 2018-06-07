import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
分类(通过tensorboard对train loss和test loss)
"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def get_accuracy(test_x,test_y):
    global output
    test_result = sess.run(output,feed_dict={input_x:test_x,input_y:test_y,keep_prob:1.0})
    compare_result = tf.equal(tf.argmax(test_result,1) ,tf.argmax(test_result,1))
    accuracy = tf.reduce_mean(tf.cast(compare_result,tf.float32))
    return accuracy

def add_layer(x_data,input_size,out_size,active_function=None):
    weights = tf.Variable(tf.random_normal([input_size,out_size]))
    bais = tf.Variable(tf.zeros([1,out_size]) + 0.01)
    z = tf.add(tf.matmul(x_data,weights),bais)
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

loss = -tf.reduce_mean(input_y * tf.log(output))
# 统计损失
tf.summary.histogram("loss",loss)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    train_board = tf.summary.FileWriter("tensorboard/train",sess.graph)
    test_board = tf.summary.FileWriter("tensorboard/test", sess.graph)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={input_x:batch_xs,input_y:batch_ys,keep_prob:0.5})
        if i % 100 == 0:
            # 添加统计信息到tensorboard
            train_board.add_summary((sess.run(merged, feed_dict={input_x: batch_xs, input_y: batch_ys, keep_prob: 1.0})),i)
            test_board.add_summary(sess.run(merged,feed_dict={input_x:mnist.test.images,input_y:mnist.test.labels,keep_prob:1.0}),i)
            print(sess.run(loss,feed_dict={input_x:mnist.test.images,input_y:mnist.test.labels,keep_prob:1.0}))

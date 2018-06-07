import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def get_accuracy(test_x,test_y):
    global predictor
    test_predictor = sess.run(predictor,feed_dict={x_input:test_x,y_input:test_y,keep_prob:1.0})
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_y,1),tf.argmax(test_predictor,1)),tf.float32))
    return accuracy

def add_fclayer(inputs,input_size,out_size,active_funcatino=None,keep_prob=None):
    # truncated_normal与random_normal类似，但是truncated_normal会指定区间的数字
    Weights = tf.Variable(tf.truncated_normal([input_size,out_size],stddev=0.1))
    # Bais用数组表示
    Bais = tf.Variable(tf.constant(0.01,shape=[out_size]))
    # 使用广播
    Z = active_funcatino(tf.matmul(inputs,Weights) + Bais)
    if keep_prob is not None:
        return tf.nn.dropout(Z,keep_prob=keep_prob)
    else:
        return Z

def add_convlayer(inputs,filter_size=None,filter_strides=None,filter_padding=None,pool_size=None,pool_strides=None,pool_padding=None):
    Weights = tf.Variable(tf.truncated_normal(filter_size,stddev=0.1))
    # Bais用数组表示
    Bais = tf.Variable(tf.constant(0.1,shape=[filter_size[-1]]))
    conv_layer = tf.nn.conv2d(inputs,Weights,strides=filter_strides,padding=filter_padding)
    # 使用广播
    Z = tf.nn.relu(conv_layer + Bais)
    pool_layer = tf.nn.max_pool(Z,ksize=pool_size,strides=pool_strides,padding=pool_padding)
    return pool_layer

x_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x_input,[-1,28,28,1])

# output size 14x14x32
conv_layer1 = add_convlayer(x_image,
                            filter_size=[5,5,1,32],
                            filter_strides=[1,1,1,1],
                            filter_padding='SAME',
                            pool_size=[1,2,2,1],
                            pool_strides=[1,2,2,1],
                            pool_padding='SAME')
# output size 7x7x64
conv_layer2 = add_convlayer(conv_layer1,
                            filter_size=[5,5,32,64],
                            filter_strides=[1,1,1,1],
                            filter_padding='SAME',
                            pool_size=[1,2,2,1],
                            pool_strides=[1,2,2,1],
                            pool_padding='SAME')

conv_out = tf.reshape(conv_layer2,[-1,7*7*64])
fc_layer1 = add_fclayer(conv_out,
                        input_size=7*7*64,
                        out_size=1024,
                        active_funcatino=tf.nn.relu,
                        keep_prob=keep_prob)

predictor = add_fclayer(fc_layer1,
                        input_size=1024,
                        out_size=10,
                        active_funcatino =tf.nn.softmax,
                        keep_prob=None)

loss = -tf.reduce_mean(y_input * tf.log(predictor))
optimize = tf.train.AdamOptimizer(1e-5).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimize,feed_dict={x_input:batch_xs,y_input:batch_ys,keep_prob:0.5})
        if i % 100 == 0:
            print(sess.run(loss,feed_dict={x_input:mnist.test.images,y_input:mnist.test.labels,keep_prob:1.0}))
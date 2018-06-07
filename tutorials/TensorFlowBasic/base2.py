import tensorflow as tf
import  numpy as np

"""
常量、变量赋值、placeholder
"""

# 定义常量
data1 = tf.constant(10)
data2 = tf.constant([10,10])
data2 = tf.constant([[10,10],
                     [10,10]])

X_data = tf.Variable(tf.ones([10,10]),dtype=tf.float32)
# 更新变量的值
tf.assign(X_data,tf.random_normal([10,10],dtype=tf.float32))

# placeholder类似于变量,但是可以在使用的时候赋值
ph1 = tf.placeholder(tf.float32,[None,10]) # [None,10] N行,10列

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(X_data))
    # placeholder对象赋值的时候，必须为所有的placeholder指定值
    # 不能赋予tensor对象,比如:tf.randome_normal([2,10])
    print(sess.run(ph1,feed_dict={ph1:np.random.randn(2,10)}))
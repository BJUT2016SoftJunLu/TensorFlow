import  tensorflow as tf
import numpy as np

"""
通过梯度下降更新Weight和Bais
"""

# 定义变量,定义的时候必须指定值
X_data = tf.Variable(np.random.randn(1000),dtype=tf.float32,name="X_data")
Y_data = 0.3 * X_data + 0.1

Weight = tf.Variable(np.random.randn(1),dtype=tf.float32,name="Weight")
Bais = tf.Variable(np.random.randn(1),dtype=tf.float32,name="Bais")

predict = tf.add(tf.multiply(Weight,X_data),Bais)

# 定义算是函数_平方根误差
loss = tf.reduce_mean(tf.square(tf.subtract(Y_data,predict)))
optimizer= tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(optimizer)
    print(sess.run(Weight))

import tensorflow as tf
import  numpy as np


"""
使用神经网络进行回归训练
"""

# np.linspace 返回间隔相同的数列
# [:, np.newaxis] 表示每一个数字一行,[np.newaxis,:] 表示所有数据一行
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义一层网络
def add_layer(x_data,input_size,out_size,active_function=None):
    weights = tf.Variable(tf.random_normal([input_size,out_size]))
    bais = tf.Variable(tf.zeros([1,out_size]) + 0.01)
    z = tf.add(tf.matmul(x_data,weights),bais)
    if active_function is None:
        return z
    else:
        return active_function(z)



xph = tf.placeholder(tf.float32,[None,1])
yph = tf.placeholder(tf.float32,[None,1])

# 将数据输入到神经网络
layer1 = add_layer(xph,1,10,tf.nn.relu) # 第一层10个神经元
predictor =  add_layer(layer1,10,1,active_function=None)     # 输出成1个神经元

# reduction_indices  None:所有元素求平均,0:每一列求平均值  1:每一行求平均值
loss = tf.reduce_mean(tf.square(yph - predictor),reduction_indices=None)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init= tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # 迭代1000次
    for i in range(3000):
        sess.run(optimizer,feed_dict={xph:x_data,yph:y_data})
        # 每50次查看一次损失变化
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xph:x_data,yph:y_data}))

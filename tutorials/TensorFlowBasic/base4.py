import tensorflow as tf
import numpy as np

"""
tensorboard
"""

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

def add_layer(x_data,input_size,output_size,Active_funcation=None):
    with tf.name_scope("layer"):
        with tf.name_scope("Weight"):
            Weights = tf.Variable(tf.random_normal([input_size,output_size]))
            tf.summary.histogram("Weight",Weights)
        with tf.name_scope("Bais"):
            Bais = tf.Variable(tf.zeros([1,output_size]) + 0.01)
            tf.summary.histogram("Bais",Bais)
        with tf.name_scope("Z"):
            Z = tf.add(tf.matmul(x_data,Weights),Bais)
        if Active_funcation is None:
            return Z
        else:
            return Active_funcation(Z)

with tf.name_scope("inputs"):
    with tf.name_scope("x_input"):
        input_placeholder  = tf.placeholder(tf.float32,[None,1],name="x_input")
    with tf.name_scope("y_input"):
        output_placeholder = tf.placeholder(tf.float32,[None,1],name="y_input")


hidden_layer1 = add_layer(input_placeholder,1,10,tf.nn.relu)
output_layer =  add_layer(hidden_layer1,10,1,Active_funcation=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(output_placeholder - output_layer),reduction_indices=0)
    tf.summary.histogram("loss",loss)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    # 保存tensorboard到文件
    writer = tf.summary.FileWriter("tensorboard",sess.graph)
    for i in range(1000):
       sess.run(optimizer,feed_dict={input_placeholder:x_data,output_placeholder:y_data})
       if i % 50 == 0:
           # 生成Weights,Bais的统计结果，并保存到文件
           summary = sess.run(merged,feed_dict={input_placeholder:x_data,output_placeholder:y_data})
           writer.add_summary(summary,i)
           print(sess.run(loss,feed_dict={input_placeholder:x_data,output_placeholder:y_data}))
import  tensorflow as tf

# 通过python命令执行py文件的时候传入参数
FLAGS = tf.app.flags.FLAGS
# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string("str_name",'','descrip1')
tf.app.flags.DEFINE_integer('int_name', 10,"descript2")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")

print(FLAGS.str_name)
print(FLAGS.int_name)
print(FLAGS.bool_name)

x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
# 返回一个Tensor对象
tf.stack([x, y, z])             # [[1, 4], [2, 5], [3, 6]]
tf.stack([x, y, z], axis=1)     # [[1, 2, 3], [4, 5, 6]]

v1 = tf.Variable(tf.truncated_normal([3,4,5],stddev=0.1),dtype=tf.float32)

# 根据axis的值进行维度拆分 axis = 0 表示按照行拆分  axis = 1 表示按照列拆分
# stack([tensors],axis=0) 每一个tensorszuo'w'i
# v2 = tf.unstack(v1,axis=1)
# transpose根据指定顺序排序维度 (默认是进行转置)
# v2 = tf.transpose(v1,[2,0,1])
# -1会自动根据合适的情况推断出具体的数字
v2 = tf.reshape(v1,[4,3,5])


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(v2))
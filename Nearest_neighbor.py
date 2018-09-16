# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#导入数据集
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#批量测试数据
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

# 占位符
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])
#计算距离总和为一个数距离，add函数是会逐行进行加操作的
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
#获取最小值，既返回最小向量的标号
pred = tf.arg_min(distance, 0)
accuracy = 0.
#初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xte)):
        #数据读入
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
#计算精度
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
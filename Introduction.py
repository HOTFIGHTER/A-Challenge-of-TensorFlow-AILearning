# coding=utf-8
import tensorflow as tf
#设置固定大小的文本（常量张量）
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
#运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
print(sess.run(hello))
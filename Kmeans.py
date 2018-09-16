# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#输入数据
full_data_x = mnist.train.images
# 训练次数
num_steps = 50
# 批大小
batch_size = 1024
# 分类数
k = 25
# 每张图28*28
num_features = 784
num_classes = 10

# 占位x
X = tf.placeholder(tf.float32, shape=[None, num_features])
# 占位y
Y = tf.placeholder(tf.float32, shape=[None, num_classes])
# Kmean分类
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
training_graph = kmeans.training_graph()
if len(training_graph) > 6:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph
#输入变量对应的团簇id
cluster_idx = cluster_idx[0]
#求取平均向量
avg_distance = tf.reduce_mean(scores)
#初始化向量
init_vars = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("cluster_idx",idx)
        print("Step %i, Avg Distance: %f" % (i, d))
#k个质心载入分类，分类对应的团簇加上对应labal
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
print(counts)
#选择最大的count置于map中
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)
#按照组内顺序返回cluster_idx
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
#cast类型转换，计算准确率
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
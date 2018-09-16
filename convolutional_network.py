# coding=utf-8
#卷积神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
#学习率
learning_rate = 0.001
#迭代步数
num_steps = 2000
batch_size = 128
num_input = 784
#分类数
num_classes = 10
dropout = 0.25

def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    #tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量
    with tf.variable_scope('ConvNet', reuse=reuse):
        #输入一个字典
        x = x_dict['images']
        #tf.reshape(tensor, shape, name=None)
        #函数的作用是将tensor变换为参数shape的形式。
        #其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。
        #[Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        #具有32个输出通道和5为卷积窗大小，的卷积层（relu激活函数）
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        #最大池化操作
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)

        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        #把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量
        fc1 = tf.contrib.layers.flatten(conv2)
        #全连接层（dense）输出1024
        fc1 = tf.layers.dense(fc1, 1024)
        #一种防止神经网络过拟合的手段。
        #随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果，只有在训练的时候使用
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        #全连接层（dense）
        out = tf.layers.dense(fc1, n_classes)
    return out

def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)
    #计算列方向的最大值
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    #计算交叉熵之前，通常要用到softmax层来计算结果的概率分布
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})
    return estim_specs

model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
#训练模型
model.train(input_fn, steps=num_steps)
#评价模型
#定义评价的输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)
print("Testing Accuracy:", e['accuracy'])
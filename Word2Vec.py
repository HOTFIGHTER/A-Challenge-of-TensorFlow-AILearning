# coding=utf-8
import tensorflow as tf
import os
import urllib.request
import zipfile
import numpy as np
import collections
import random

learning_rate = 0.01
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

#最大单词数
max_vocabulary_size = 50000
#最小出现数
min_occurrence = 10
#窗口
skip_window = 3
#重用词生存标签次数
num_skips = 2
#内嵌向量维度,防止one-hot编码
embedding_size = 200
num_sampled = 64

eval_words = [b'five',b'of', b'going', b'hardware', b'american', b'britain']

url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

#用UNK标记构建字典并替换稀有词
count = [('UNK', -1)]
#检索最常用词，按出现频率从大到小排列并赋值给count
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
for i in range(len(count) - 1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)#低于最小出现数踢出
    else:
        break
#计算词汇数
vocabulary_size = len(count)
#新建word2vec的字典
word2id = dict()
#枚举
for i, (word, _)in enumerate(count):
    word2id[word] = i #用于标记词汇的编号
    print("word2id:",word)
data = list()
unk_count = 0
#在语境中进行操作
for word in text_words:
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))
print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])

data_index = 0
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    #data为词汇转换后的编号，数值化
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):#便利batch_size
        #context_words为非本词的窗口区域
        context_words = [w for w in range(span) if w != skip_window]
        #随机从context_words中选择num_skips个数
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]#现在对应的词汇
            labels[i * num_skips + j, 0] = buffer[context_word]#窗口区对应的上下文
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

#占位符
X = tf.placeholder(tf.int32, shape=[None])
#占位符
Y = tf.placeholder(tf.int32, shape=[None, 1])

#算法模型部分
with tf.device('/cpu:0'):
   #创建正态分布的vocabulart_size*embedding_size向量
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
   #根据输入找到对应的相连
    X_embed = tf.nn.embedding_lookup(embedding, X)
    #初始化权重
    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    #初始化偏差
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
#计算损失函数
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
#向量相乘
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)

# 初始化默认向量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    x_test = np.array([word2id[w] for w in eval_words])

    average_loss = 0
    for step in range(1, num_steps + 1):
        #输入和输出值
        batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
        # 运行训练
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
            print("Step " + str(step) + ", Average Loss= " + \
                  "{:.4f}".format(average_loss))
            average_loss = 0

        #
        if step % eval_step == 0 or step == 1:
            print("Evaluation...")
            sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
            for i in range(len(eval_words)):
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)
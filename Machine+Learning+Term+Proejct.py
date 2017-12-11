
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd 
from pandas import DataFrame,Series 


# In[2]:


def MinMaxScaler(data): #정규화
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-5)


# In[3]:


xy = np.loadtxt('poker-hand-training-true.csv', delimiter=',', dtype=np.float32)
df=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data")

xy = MinMaxScaler(xy)
x_data = xy[0:25010 , 0:10]
y_data = xy[0:25010,10:11]
# parameters
# training_epochs = 20 #필요X
# batch_size = 100 #필요X 데이터가 너무 많은 것만 쓰면 됨
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 10])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([10, 30]), name='weight')
# b = tf.Variable(tf.random_normal([30]), name='bias')
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([10, 40]), name='weight1')
    b1 = tf.Variable(tf.random_normal([40]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([40, 50]), name='weight2')
    b2 = tf.Variable(tf.random_normal([50]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([50, 30]), name='weight3')
    b3 = tf.Variable(tf.random_normal([30]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    w3_hist = tf.summary.histogram("weights3", W2)
    b3_hist = tf.summary.histogram("biases3", b2)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_normal([30, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    w4_hist = tf.summary.histogram("weights4", W2)
    b4_hist = tf.summary.histogram("biases4", b2)
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)


# layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# W3 = tf.Variable(tf.random_normal([40, 50]))
# b3 = tf.Variable(tf.random_normal([50]))
# layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# W4 = tf.Variable(tf.random_normal([50, 1]))
# b4 = tf.Variable(tf.random_normal([1]))
# # Hypothesis (using softmax)


# In[4]:


df=df.apply(MinMaxScaler)
df.ix[:,1:].values


# In[ ]:



# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost_summ = tf.summary.scalar("cost", cost)
train = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)
prediction = tf.argmax(hypothesis, 1)


# In[ ]:



# Launch graph
with tf.Session() as sess:
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)  # Show the graph
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    for step in range(14000):
        sess.run(train, feed_dict=feed)
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))


    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_data, Y: y_data}))


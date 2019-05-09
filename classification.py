import tensorflow as tf
import numpy as np
import write as write
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

data_train=np.loadtxt('train4.txt',dtype='float32',delimiter=',')
data_test=np.loadtxt('test4.txt',dtype='float32',delimiter=',')
def label_change(before_label):
    label_num=len(before_label)
    change_arr=np.zeros((label_num,6))
    for i in range(label_num):
        change_arr[i,int(before_label[i])]=1
    return change_arr

def train(data_train):
    X_train=data_train[:,:7]
    Y_train=label_change(data_train[:,-1])
    return X_train,Y_train

X_train,Y_train,=train(data_train)

def test(data_test):
    X_test=data_test[:,:7]
    Y_test=label_change(data_test[:,-1])
    return X_test,Y_test

X_test,Y_test,=test(data_test)

def multiplayer_perception(x, weight, bias):
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    #layer1 = tf.nn.tanh(layer1)
    layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
    layer2 = tf.nn.softmax(layer2)

    out_layer = tf.add(tf.matmul(layer2, weight['out']), bias['out'])

    return out_layer





if __name__ == "__main__":
    # 定义神经网络的参数
    learning_rate = 0.01  # 学习率
    training_step = 100000000  # 训练迭代次数
    testing_step = 1  # 测试迭代次数
    display_step_test = 600
    display_step = 100  # 每多少次迭代显示一次损失
    n_input = 7
    n_hidden_1 = 10
    n_hidden_2 = 10
    n_class=6

    logs_path = '/tmp/tensorflow_logs/example/'
    # 定义输入和输出
    x = tf.placeholder(tf.float32, shape=(None, 7), name="X_train")
    y = tf.placeholder(tf.float32, shape=(None, 6), name="Y_train")
    # 定义模型参数
    weight = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="weights_h1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="wights_h2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_class]))
        }
        #tf.summary.histogram('layer1/weights',weight[0])
    bias = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_class]))
        }
        #tf.summary.histogram('layer1/bias', bias[0])
    # 定义神经网络的前向传播过程
    #Model = tf.nn.sigmoid(tf.matmul(x,weight) + bias)

    Model = multiplayer_perception(x, weight, bias)
    #Model = tf.nn.relu(tf.matmul(x,weight) + bias)
    Model = tf.nn.softmax(Model)
    """
    对模型进行优化，将Model的值加0.5之后进行取整，
    方便测试准确率(若Model>0.5则优化后会取整为1，反之会取整为0)
    """
    #model = Model + 0.5
    #model = tf.cast(model, tf.int32)
    #y_ = tf.cast(y, tf.int32)
    # Dropout操作：用于防止模型过拟合
    keep_prob = tf.placeholder(tf.float32)
    Model_drop = tf.nn.dropout(Model, keep_prob)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(Model), reduction_indices=[1]))
    cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(Model, 1e-10,1.0)) + (1-y) * tf.log(tf.clip_by_value(1 - Model, 1e-10, 1.0)))

    """
    优化函数
    即反向传播过程
    主要测试了Adam算法和梯度下降算法，Adam的效果较好
    """
    # 优化器：使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # 优化器：梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # 加载数据and数据预处理

    # 加载
    #training_data, test_data = load_data()
    # 训练集
    #X_train,Y_train = train(data_train)
    #X_train = training_data[:, :7]
    #Y_train = training_data[:, 7:8]
    # 测试集
    #X_test, Y_test = train(data_test)
    #X_test = test_data[:, :7]
    #Y_test = test_data[:, 7:8]
    # X_test_Mn = StandardScaler().fit_transform(X_test)
    b = MinMaxScaler()
    X_test_cen = b.fit_transform(X_test)
    # 1、标准化
    # X_train_Mn = StandardScaler().fit_transform(X_train)
    # 2、正则化 norm为正则化方法：'l1','l2','max'
    # X_train_nor = Normalizer(norm='max').fit_transform(X_train)
    # 3、归一化(centering)
    a = MinMaxScaler()
    X_train_cen = a.fit_transform(X_train)
    # 计算准确度
    correct_prediction = tf.equal(tf.argmax(Model,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求所有correct_prediction的均值
    #correct_prediction = tf.equal(tf.argmax(Model,1),tf.argmax(y,1))
    # 创建会话运行TensorFlow程序
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化变量
        init = tf.global_variables_initializer()
        #saver.restore(sess, '/tmp/tensorflow_logs/log/save_net.ckpt')
        sess.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        # 训练测试集
        for i in range(training_step):
            # 训练模型运行语句（采用矩阵运算将训练时间减少至十几秒）
            sess.run(optimizer, feed_dict={x: X_train_cen, y: Y_train, keep_prob: 0.5})
            # 每迭代1000次输出一次日志信息
            # display = (i % 10)
            if i % display_step == 0:
                # 输出交叉熵之和
                total_cross_entropy_train = sess.run(cross_entropy, feed_dict={x: X_train_cen, y: Y_train})
                print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy_train))
                # 输出准确度
                # 每10轮迭代计算一次准确度
                accuracy_rate = sess.run(accuracy, feed_dict={x: X_train_cen, y: Y_train, keep_prob: 1.0})
                print('第' + str(i) + '轮,Training的准确度为：' + str(accuracy_rate))
                if accuracy_rate > 0.95:
                #if total_cross_entropy_train <0.1:
                    a1 = open("test-result", "a")
                    a1.write(str(i+100) + " ")
                    a1.close()
                    break
                #writer.add_summary(result, i)
                saver.save(sess, '/tmp/tensorflow_logs/log/save_net.ckpt')
        # 测试数据集

        for i in range(testing_step):
            # 通过选取样本训练神经网络并更新参数
            sess.run(optimizer, feed_dict={x: X_test_cen, y: Y_test})

            # 每迭代1000次输出一次日志信息
            # display1 = (i % 10)
            if i % display_step_test == 0:
                # 计算所有数据的交叉熵
                total_cross_entropy_test = sess.run(cross_entropy, feed_dict={x: X_test_cen, y: Y_test})
                print("After %d testing step(s),cross entropy on all data is %g" % (i, total_cross_entropy_test))
                # if (display1 == 0) and (i<len(X_test)):
                accuracy_rate1 = sess.run(accuracy, feed_dict={x: X_test_cen, y: Y_test, keep_prob: 1.0})
                a1 = open("test-result", "a")
                a1.write(str(accuracy_rate1) + "\n")
                a1.close()
                print('第' + str(i) + '轮,Testing的准确度为：' + str(accuracy_rate1))
                # 输出预测的结果和期望的结果
        pred_Y_test = sess.run(Model, feed_dict={x: X_test_cen})
        print('测试结果如下：')
        for pred, real in zip(pred_Y_test, Y_test):
            print(pred, real)
        print('\n\n')
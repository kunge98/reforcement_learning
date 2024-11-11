"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
tf.set_random_seed(1)

# 莫凡老师代码为1.X版本，不兼容，所以只了解代码的思路即可

# Deep Q Network  是一种离线学习方法 off-policy
class DeepQNetwork:

    # n_actions多少个状态；n_features接受多少个observation；learning_rate学习率，reward_decay就是伽马
    # e_greedy是爱不惜龙，90%的时候选择最大的动作；replace_target_iter隔多少步把q现实的参数替换掉；memory_size记忆的容量大小；batch_size批次
    # e_greedy_increment不断的增大该值，即减小随机数值的大小；output_graph图像显示
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.9,
                e_greedy=0.9,replace_target_iter=300,memory_size=500,batch_size=32,
                e_greedy_increment=None,output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 记录学习的步数
        self.learn_step_counter = 0

        # 创建一个全为0的记忆库 [s, a, r, s_]
        # n_features接受多少个observation，一个transition包括当前状态，动作，奖励，下一状态，即 两个状态 + 动作 + 奖励
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2)))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    # 建立神经网络
    def _build_net(self):


        # 1.建立训练网络

        # 两个输入
        # 一个是状态，通过神经网络得到q估计,q估计用的是新参数
        # 另一个输入是q_target，q现实，  q现实用的是旧参数，两个q值的差得到loss
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        # eval_net训练网络
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            # c_names, n_l1, w_initializer, b_initializer = \
            #     ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
            #     tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # 建立每层的默认参数
            # c_names为q估计的参数都放在此集合中，n_l1第一层有多少神经元，w为权重，b为偏置
            c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)

            # 第一层 collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二层 collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # 计算q估计和q现实的差距，loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # 选择最小的loss，和优化器进行训练
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



        # 2.建立target——network网络

        # 输入下一个状态
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input

        # target_net网络
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            # c_names中保存的是q现实的参数（旧参数）
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一层 collections is used later when assign to target net
            # 赋值权重和偏置，计算l1
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # 第一层 collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2


    # 存储transition记忆，mermory库大小有限，思路是先存满，然后用最新存储的transition替换最老的
    def store_transition(self, s, a, r, s_):

        # 如果没有memory_counter属性
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 在数组的最后水平插入一个transition
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 选择动作
    def choose_action(self, observation):
        # 输入进来的observation是一维，增加batch维度使用神经网络
        observation = observation[np.newaxis, :]

        # 如果小于爱不惜龙，选择最大的动作值，否则随机选择动作值
        if np.random.uniform() < self.epsilon:
            # 将该状态的下的observation都放入q_eval网络中运行，将actions_value存储所有输出值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习
    def learn(self):
        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从 memory 中随机抽取 batch_size 这么多记忆
        # 它是一个batch一个batch的训练的 不是说每一步就训练哪一步
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            })

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # batch_memory就是之前储存下来的记忆，根据不同的需要有所改变
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
        比如在:
            记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
            记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        所以 (q_target - q_eval) 就变成了:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
        所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
        我们只反向传递之前选择的 action 的值,
        """

        # 训练 eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  # 记录 cost 误差

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    # 展示学习曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()




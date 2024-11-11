import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        # 是否输出tensorboard文件
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())


    # 创建网络
    def _build_net(self):
        with tf.name_scope('inputs'):
            # 接收 observation
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # 接收在这个回合选过的actions
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # 接收每个回合state-action 所对应的value（通过reward计算）
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # 第一层
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            # 输出个数
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # 第二层
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # 使用softmax转换成概率
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        # 计算‘loss’
        with tf.name_scope('loss'):
            # 最大化最终总奖励 (log_p * R) ，tensorflow中只有最小化操作，所以前面加一个负号
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # vt = reward + 衰减未来的reward 从来引导参数进行梯度下降
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    # 不再用Q-value来选取动作，而是根据概率选择动作
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action


    # 存储transition
    # 将观测值、动作、奖励存储到列表中，因为本行回合完毕之后要进行清空列表，然后存储下一回合的数据，所以会在learn方法中进行清空列表的动作
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)


    # 学习更新参数
    def learn(self):
        # 衰减，并标准化这回合的reward，定义的变量说明了一切
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # 对一个episode进行训练
        # self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        # 清空列表内容
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        # 返回这一回合的state-action 的 value
        return discounted_ep_rs_norm



    # 衰减回合的reward
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs




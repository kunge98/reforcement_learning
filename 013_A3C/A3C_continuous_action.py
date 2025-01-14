import multiprocessing
import threading
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

# 超参数
GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
# 创建的本地网络的数量和自己电脑的cpu的数量一样
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 5000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01

# actor的学习率
LR_A = 0.0001
# critic的学习率
LR_C = 0.001

GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):

    # 初始化
    def __init__(self, scope, globalAC=None):

        # 如果是GLOBAL_NET_SCOPE则创建global network
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # 公共网络只关心Actor和Critic的参数
                self.a_params, self.c_params = self._build_net(scope)[-2:]

        # 否则建立local network，计算loss等
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                # TD_error计算
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # 取动作的正态分布
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    # 增加探索性
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    # 取负号反向操作
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                # 使用本地参数来选择动作
                with tf.name_scope('choose_a'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    # 把最新的参数更新到本地网络中去
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

                with tf.name_scope('push'):
                    # 把最新的参数（梯度）用在公共网络中更新
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    # 创建网络
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    # 更新主网络
    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    # work运行的时候使用公共网络的参数
    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # 选择动作
    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


# 主网络的副本
class Worker(object):

    # 初始化
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    #
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1

        # 存储状态、动作、奖励的空列表（缓存）
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                # 存入缓冲中
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                # 并标准化
                buffer_r.append((r+8)/8)

                # 更新公共网络并且发布到本地网络中
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    # 如果是terminal，则将未来的期望设置为0
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # RMSPropOptimizer为优化器
        # 公共网络OPT_A，OPT_C
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        # 其中公共网络只是包含了一些参数，不包括训练
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []

        # 创造worker
        for i in range(N_WORKERS):
            # worker的名字
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, GLOBAL_AC))

    # 多线程调度器
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # 输出图像
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    # 创建多线程，开始并行
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()


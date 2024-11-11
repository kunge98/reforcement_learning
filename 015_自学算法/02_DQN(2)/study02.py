# 复现DQN

import random
import numpy as np
import gym
# import imageio  # write env render to mp4
# import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env
'''

# 复现Deep-Q-Network（DQN）

class DQN:

    # 初始化
    def __init__(
            self,
            env,
            memory_cap=1000, # 换进容量
            time_steps=3, # 时间步长为3
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995, # 爱不惜龙每次衰减0.005
            epsilon_min=0.01, # 爱不惜龙最小为0.01
            learning_rate=0.005,
            batch_size=32,
            tau=0.125 # 超参数，在target_update更新函数中应用到计算target网络的参数
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape  # 当前状态的形状，就是当前观测到的状态的维度
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))  # 存储当前状态使用一个全为0的一维数组
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 贪婪策略爱不惜龙
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # 贪婪因子衰减
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target目标网络的更新
        self.model = self.create_model()  # 创建Q估计网络模型
        self.target_model = self.create_model()  # 创建Q现实网络模型
        self.target_model.set_weights(self.model.get_weights()) # 设置target目标网络的权重（从Q估计的哪里获取）

        self.summaries = {} # 存放数据，在replay函数中得到应用

    # 创建模型函数
    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape[0] * self.time_steps, activation="relu"))
        model.add(Dense(16, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        # 输出的是当前环境下所有可能的动作
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    # 更新状态
    def update_states(self,new_state):
        # roll作用就是数组中最后面的向前挤，挤掉最前面的最老的状态，空出数组的最后一个位置
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        # 把最新的状态存在数组的最后一个位置
        self.stored_states[-1] = new_state

    # 选择动作，test为false意思是在非测试环境下
    def act(self, test=False):
        # 将存放状态的数组变成一维的
        states = self.stored_states.reshape((1, self.state_shape[0] * self.time_steps))
        # 爱不惜龙进行更新，因为在开始设置的是1，所以在选择动作之前将其减小，以免选择动作的时候值为1就100%选中了
        self.epsilon *= self.epsilon_decay
        # 选择最大的那个爱不惜龙
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # 当进行测试的时候爱不惜龙的值为0.01，不是测试环境爱不惜龙的值为当前的值
        epsilon = 0.01 if test else self.epsilon
        # 求出状态下所有动作的Q值
        q_values = self.model.predict(states)[0]
        # 存放Q值最大的那个 -- （Q估计）
        self.summaries['q_val'] = max(q_values)
        # 如果产生的随机数大于爱不惜龙
        if np.random.random() < epsilon:
            # 随机返回一个动作，如果小于那么就选择动作价值最大的那个
            return self.env.action_space.sample()
        return np.argmax(q_values)

    # 存储记忆
    def remember(self, state, action, reward, new_state, done):
        # 在记忆库的最后append，添加一个transition
        self.memory.append([state, action, reward, new_state, done])

    # 经验回放
    def replay(self):

        # 如果当前的记忆库的大小比一个批次的容量都还小，那么就返回接着去存储记忆，说明存储的记忆还不够
        if len(self.memory) < self.batch_size:
            return

        # 记忆库存储完成，随机抽样选择一个批次的样本
        samples = random.sample(self.memory, self.batch_size)
        # 通过函数反向解压样本samples分别得到状态，动作，奖励，下一状态，是否完成
        states, action, reward, new_states, done = map(np.asarray, zip(*samples))
        # 将状态和下一状态转变成array数组并重新定义维度
        batch_states = np.array(states).reshape(self.batch_size, -1)
        batch_new_states = np.array(new_states).reshape(self.batch_size, -1)
        # 使用一组状态对Q-target目标网络进行预测估计
        batch_target = self.target_model.predict(batch_states)
        # 使用一组 下一状态 对target目标网络进行预测估计并选择最大的那个
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        # 计算Q-target，当前奖励 + 衰减因子 * gamma * maxQ
        batch_target[range(self.batch_size), action] = reward + (1 - done) * q_future * self.gamma
        # 运行模型
        hist = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        # np.mean求数组中loss的均值，赋值在summaries数组中，列名为loss
        self.summaries['loss'] = np.mean(hist.history['loss'])

    # target目标网络的更新
    def target_update(self):
        # weights是从Q-估计网络中获取的权重
        weights = self.model.get_weights()
        # target_weights是从Q-现实网络中获取的权重
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            # Q-target网络权重的更新要结合两个不同网络的权重分别乘不同的参数来得到新的target网络的权重
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        # 返回的是计算过后的新的权重并set设置权重给target网络
        self.target_model.set_weights(target_weights)


    # 进行训练
    def train(self, max_episodes, max_steps=1000): # save_freq = 10

        # # 当前时间
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # # 训练的日志文件
        # train_log_dir = 'logs/DQN_basic_time_step{}/'.format(self.time_steps) + current_time
        # # 对改文件进行写入操作
        # summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = True, 0, 0, 0, 0
        # episode小于最大循环次数的时候就执行，否则代表到达了最大循环次数，结束循环并保存模型
        while episode < max_episodes:
            # 到达了最大的步数就停止，输出reached max steps，保存模型
            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                # self.save_model("dqn_basic_maxed_episode{}_time_step{}.h5".format(episode, self.time_steps))

            # 默认done是完成的状态
            if done:
                # with summary_writer.as_default():
                #     tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                #     tf.summary.scalar('Main/episode_steps', steps, step=episode)

                self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
                print("当前迭代次数：episode{}，当前步数为：step{},当前奖励reward：{}".format(episode, steps,total_reward))

                # if episode % save_freq == 0:  # save model every n episodes
                #     self.save_model("dqn_basic_episode{}_time_step{}.h5".format(episode, self.time_steps))

                # 如果done是完成的话，说明本次过程结束，将一切参数都改变
                # done改回原来的未完成状态，当前状态由环境重置得到一个新状态，步数和总奖励清零
                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                # 由update_states函数将新得到的状态存入数组中由此更新
                self.update_states(cur_state)
                # 循环次数加一
                episode += 1

            # 调用act函数选择动作
            action = self.act()  # model determine action, states taken from self.stored_states
            # 在环境中执行动作得到新状态、奖励、是否完成等
            new_state, reward, done, _ = self.env.step(action)

            # modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
            prev_stored_states = self.stored_states
            # 将新状态存入数组中
            self.update_states(new_state)
            # 将transition存入记忆库中
            self.remember(prev_stored_states, action, reward, self.stored_states, done)  # add to memory
            # 经验回放
            self.replay()  # iterates default (prediction) model through memory replay
            # 对target目标网络进行参数更新
            self.target_update()

            # 总奖励、步数、循环次数都累加
            total_reward += reward
            steps += 1
            epoch += 1

            # 可视化Tensorboard更新
            # with summary_writer.as_default():
            #     if len(self.memory) > self.batch_size:
            #         tf.summary.scalar('Stats/loss', self.summaries['loss'], step=epoch)
            #     tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
            #     tf.summary.scalar('Main/step_reward', reward, step=epoch)

            # summary_writer.flush()

        # 循环次数到达最大的时候（循环结束），保存模型
        # self.save_model("dqn_basic_final_episode{}_time_step{}.h5".format(episode, self.time_steps))

    # 进行测试
    # def test(self, render=True, fps=30, filename='test_render.mp4'):
    #     # 重置环境，得到状态。done未完成，奖励清零
    #     cur_state, done, rewards = self.env.reset(), False, 0
    #     video = imageio.get_writer(filename, fps=fps)
    #     while not done:
    #         action = self.act(test=True)
    #         new_state, reward, done, _ = self.env.step(action)
    #         self.update_states(new_state)
    #         rewards += reward
    #         if render:
    #             video.append_data(self.env.render(mode='rgb_array'))
    #     video.close()
    #     return rewards


    # 保存这个模型

    # def save_model(self, fn):
    #     # 保存模型，文件的后缀名为.h5
    #     self.model.save(fn)

    # 加载模型
    # def load_model(self, fn):
    #     # 加载后缀名为.h5的文件
    #     self.model = tf.keras.models.load_model(fn)
    #     self.target_model = self.create_model()
    #     self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500
    # 通过DQN类创建了一个对象
    dqn_agent = DQN(env, time_steps=4)
    # dqn_agent.load_model("basic_models/time_step4/dqn_basic_episode50_time_step4.h5")
    # rewards = dqn_agent.test()
    # print("Total rewards: ", rewards)
    dqn_agent.train(max_episodes=200)

import copy

import tensorflow as tf
import gym.envs as envs
import numpy as np
import matplotlib.pyplot as plt
import gym


# 查看可以玩的游戏
print(envs.registry.all())

env = gym.make('CartPole-v0')
env.reset()
while True:
    env.render()
    env.action_space(1)


class DQNAgent(object):

        # 初始化
        def __init__(self,_env):
            self.env = _env
            # 经验池
            self.memory = []
            # 奖励衰减
            self.gamma = 0.9
            # 控制训练的随机干涉
            # 随机干涉阈值，该值会随着训练减少
            self.epslion = 1
            # 每次随机衰减0.005
            self.epslion_decay = .995
            self.epslion_min = 0
            # 学习率
            self.learning_rate = 0.0001
            # 创建模型
            self._build_model()

        # 创建模型
        def _build_model(self):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(128,input_dim=4,activation='tanh'))
            model.add(tf.keras.layers.Dense(128,activation='tanh'))
            model.add(tf.keras.layers.Dense(128,activation='tanh'))
            model.add(tf.keras.layers.Dense(2,activation='linear'))
            model.summary()
            model.compile(loss='mse',metrics=['acc'],optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate))
            # model.commpile(loss='mse',metrics=['acc'],optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate))
            self.model = model

        # 存储经验,存进memory中
        def save_exp(self,_state,_action,_reward,_next_state,_done):
            self.memory.append((_state,_action,_reward,_next_state,_done))

        # 经验池重放，根据尺寸获取
        def train_exp(self,batch_size):
            # 确保每批返回的数量不会超出memory实际的存量，防止错误
            batches = min(batch_size,len(self.memory))
            # 从len(self.memory）中随机选出batches个数
            batches = np.random.choice(len(self.memory),batches)

            for i in batches:
                # 从经验数组中 取出相对的参数 状态，行为，奖励，即将发生的状态，结束状态
                _state, _action, _reward, _next_state, _done = self.memory[i]

                # 获取当前 奖励,如果已经是最后一步
                y_reward = _reward

                if not _done:
                    # 根据_next_state  预测取得  当前回报+未来回报*折扣因子
                    y_reward = _reward + self.gamma * np.max(self.model.predict(_next_state)[0])

                # 根据当前状态推断的行为
                _y = self.model.predict(_state)
                # print(_y,y_reward,_action)

                _y[0][_action] = y_reward

                # 训练
                self.model.fit(_state,_y,epochs=5,verbose=0)


            if self.epslion > self.epslion_min:
                self.epslion *= self.epslion_decay

        def action(self,_state):

            # 随机返回0-1的数，
            # 随着训练的增加，渐渐减少随机
            if np.random.rand() >= self.epslion:
                return self.env.action_space.sample()
            else:
                act_values = self.model.predict(_state)
                return np.argmax(act_values[0])

if __name__ == '__main__':

    # 为agent初始化gym环境参数
    envs = gym.make('CartPole-v0')
    agent = DQNAgent(envs)
    # 游戏结束规则：杆子角度为±12， car移动距离为±2.4，分数限制为最高200分

    episodes = 1000
    for e in range(episodes):
        state = envs.reset()
        state = np.reshape(state,[1,4])

        # time_t 代表游戏的每一帧 由于每次的分数是+1 或-100所以这个时间坚持的越长分数越高
        for tine_t in range(5000):
            envs.render()
            _action = agent.action(state)

            _next_state,_reward,_done,_ = envs.step(_action)
            _next_state = np.reshape(_next_state,[1,4])

            _reward = -100 if _done else _reward

            agent.save_exp(state,_action,_reward,_next_state,_done)

            state = copy.deepcopy(_next_state)

            if _done:
                print('episodes:{}/{},score:{}'.format(e,episodes,tine_t))
                break

        agent.train_exp(32)

        if(e +1) % 10 ==0:
            print('saving model')
            agent.model.save('dqn.h5')












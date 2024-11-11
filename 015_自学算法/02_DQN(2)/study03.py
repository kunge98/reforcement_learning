# 手写复现DQN
import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque


class DQN:

    # 初始化
    def __init__(self,
                 env,
                 memory_cap=1000,
                 time_steps=3,
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 learning_rate=0.005,
                 batch_size=32,
                 tau=0.125):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps,self.state_shape[0]))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.summaries = {}

    # 搭建神经网络
    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24,input_dim=self.state_shape[0] * self.time_steps,activation='relu'))
        model.add(tf.keras.layers.Dense(16,activation='relu'))
        # 输出当前环境下所有可能的动作
        model.add(tf.keras.layers.Dense(self.env.action_space.n))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    # 更新状态
    def update_state(self,new_state):
        self.stored_states = np.roll(self.stored_states,-1,axis=0)
        self.stored_states[-1] = new_state

    # 选择动作
    def action(self):
        states = self.stored_states.reshape((1,self.state_shape[0] * self.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # self.epsilon = self.epsilon * self.epsilon_decay
        epsilon = self.epsilon
        q_values = self.model.predict(states)[0]
        # 存放的是动作值最大的Q估计
        self.summaries['q_eval'] = max(q_values)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(q_values)

    # 存储记忆transition
    def remember(self,state,action,reward,new_state,done):
        self.memory.append([state,action,reward,new_state,done])

    # 经验回放
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory,self.batch_size)
        states,action,reward,new_states,done = map(np.asarray,zip(*samples))
        batch_states = np.array(states).reshape(self.batch_size,-1)
        batch_new_states = np.array(new_states).reshape(self.batch_size,-1)
        batch_target = self.target_model.predict(batch_states)
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        batch_target[range(self.batch_size),action] = reward + self.gamma * q_future * (1 - done)
        # 运行模型
        his = self.model.fit(batch_states,batch_target,epochs=1,verbose=0)
        self.summaries['loss'] = np.mean(his.history['loss'])

    # 更新target网络的参数
    def target_update(self):
        weigths = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weigths[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    # 进行训练
    def train(self,max_episodes,max_steps):
        episode,steps,epoch,total_reward,done = 0,0,0,0,True

        while episode < max_episodes:
             if steps >= max_steps:
                 print('episode{},reached max steps'.format(episode))

             if done:
                 # 创建的一维数组要以元组的形式存放
                 self.stored_states = np.zeros((self.time_steps,self.state_shape[0]))
                 print('当前迭代次数：episode{}，当前步数为：step{},当前奖励total_reward：{}'.format(episode,steps,total_reward))
                 cur_state,steps,total_reward,done = self.env.reset(),0,0,False
                 self.update_state(cur_state)
                 episode += 1

             action = self.action()
             new_state,reward,done,_ = self.env.step(action)
             prev_stored_states = self.stored_states
             self.update_state(new_state)
             self.remember(prev_stored_states,action,reward,self.stored_states,done)
             self.replay()
             self.target_update()
             total_reward += reward
             steps += 1
             epoch += 1


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500
    dqn_agent = DQN(env,time_steps=4)
    dqn_agent.train(max_episodes=200,max_steps=1000)
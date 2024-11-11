# 复现DQN

# 实际上并没有复现成功

import collections
import numpy as np
import tensorflow as tf
import gym

# 构建网络模型
class Model:

    def __init__(self,observation_n,action_dim):
        self.action_dim = action_dim
        self.observation_n = observation_n
        self._build_model()

    # 创建两层网络
    def _build_model(self):
        # Q_eval
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.observation_n)))
        model.add(tf.keras.layers.Dense(128,activation='relu',name='layer_1'))
        model.add(tf.keras.layers.Dense(128,activation='relu',name='layer_2'))
        model.add(tf.keras.layers.Dense(self.action_dim,name='layer_3'))
        # model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        model.summary()
        self.model = model
        # Q_target
        target_model = tf.keras.Sequential()
        target_model.add(tf.keras.Input(shape=(self.observation_n)))
        target_model.add(tf.keras.layers.Dense(128,activation='relu',name='layer_1'))
        target_model.add(tf.keras.layers.Dense(128,activation='relu',name='layer_2'))
        target_model.add(tf.keras.layers.Dense(self.action_dim,name='layer_3'))
        # model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        target_model.summary()
        self.model = target_model


# 算法
class DQN:

    def __init__(self,model,gamma=0.9,learning_rate=0.01):
        self.model = model.model
        self.target_model = model.target_model
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.model.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.loss_func = tf.losses.MeanSquaredError()

        # 全局迭代次数
        self.global_step = 0

        # 每隔200步就把target的参数更新一遍
        self.update_target_steps = 200


    #
    def predict(self,observation):
        return self.model.predict(observation)

    # 训练步骤
    def _train_step(self,action,feature,labels):


        with tf.GradientTape() as tape:

            # 计算Q（s，a）与target——Q的均方差，得到loss
            predictions = self.model(feature,training=True)
            enum_list = list(enumerate(action))
            # 对动作的预测值
            pred_action_value = tf.gather_nd(predictions,indices=enum_list)
            # 进行loss函数的计算
            loss = self.model.loss_func(labels,pred_action_value)
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))

    # 训练模型
    def _train_model(self,action,features,labels,epoch=1):
        for epoch in tf.range(1,epoch+1):
            self._train_step(action,features,labels)


    def learn(self,observation,action,reward,next_observation,terminal):
        # 使用DQN算法来更新self.model的value网络

        # 如果到了更新的步数，则更新target网络的参数
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()

        # 从target目标网络中获取maxQ 的数值，用于计算target——Q
        next_pred_value = self.target_model.predict(next_observation)
        best_value = tf.reduce_max(next_pred_value,axis=1)
        # 转换类型
        terminal = tf.cast(terminal,dtype=tf.float32)
        # Q_target
        target = reward + self.gamma * (1.0 - terminal) * best_value

        self._train_model(action,observation,target,epoch=1)
        # 全局步数+1
        self.global_step +=1

    # 更新target网络的权重
    def replace_target(self):
        # 预测模型的权重更新到target模型权重
        self.target_model.get_layer(name='layer_1').set_weigths(self.model.get_layer(name='layer_1').get_weights())
        self.target_model.get_layer(name='layer_2').set_weigths(self.model.get_layer(name='layer_2').get_weights())
        self.target_model.get_layer(name='layer_3').set_weigths(self.model.get_layer(name='layer_3').get_weights())


# agent
class Agent:

    def __init__(self,action_dim,algorithm,e_greed=0.9,e_greed_increasement=0,):
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.e_greed = e_greed
        self.e_greed_increasement = e_greed_increasement

    # 随机抽样,得到动作
    def sample(self,observation):
        # 产生0-1之间的数字
        sample = np.random.rand()
        if sample < self.e_greed:
            actor = self.predict(observation)
        else:
            actor = np.random.randint(self.action_dim)
        # 更新爱不惜龙参数
        self.e_greed = max(0.01,self.e_greed-self.e_greed_increasement)
        # 返回选择的动作
        return actor

    # 找出最大的那个动作
    def predict(self,observation):
        observation = tf.expand_dims(observation,axis=0)
        action = self.algorithm.model.predict(observation)
        return np.argmax(action)

# 存储记忆,经验回放
class Memory:

    def __init__(self,maxsize):
        self.buffer = collections.deque(maxlen=maxsize)

    # 存储记忆
    def append(self,exp):
        self.buffer.append(exp)

    def sample(self,batch_size):
        mini_batch = np.random.sample(self.buffer,batch_size)
        # 将这几部分分别存入空的列表中
        observation_batch,action_batch,reward_batch,next_observation_batch,terminal_batch = [],[],[],[],[]

        # 在最小的记忆批次中循环学习
        for exp in mini_batch:
            # 分别存入空列表中
            s,a,r,s_,done = exp
            observation_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_observation_batch.append(s_)
            terminal_batch.append(done)

        return np.array(observation_batch).astype('float32'),\
                np.array(action_batch).astype('int32'),\
                np.array(reward_batch).astype('float32'),\
                np.array(next_observation_batch).astype('float32'),\
                np.array(terminal_batch).astype('float32')

        # def __len__(self):
        #     return len(self.buffer)


# 超参数
# 记忆库大小
MEMORY_SIZE = 20000
# replay需要预存的一些经验，tranisition
MEMORY_WARMUP_SIZE = 200
# 训练评率
LEARN_FREQ = 5
# 训练批次大小
BATCH_SIZE = 32
# 学习率
LEARNING_RATE = 0.001
GAMMA = 0.99


def run_episode(env,algorithm,agent,rpm):
    step = 0
    total_reward = 0
    observation = env.reset()

    while True:
        step +=1
        # 随机选择一个动作
        action = agent.sample(observation)
        next_observation,reward,done = env.step(action)
        rpm.append((observation,action,reward,next_observation,done))

        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            batch_observation,batch_action,batch_reward,batch_next_observation,batch_done = rpm.sample(BATCH_SIZE)
            algorithm.learn(batch_observation,batch_action,batch_reward,batch_next_observation,batch_done)

        observation = next_observation

        total_reward += reward

        if done:
            break

    return total_reward


# 评估agent，跑5个episode，总reward求平均
def evaluate(env,agent,render=False):
    eval_reward = []
    for i in range(5):
        observation = env.reset()
        episode_reward = 0
        while True:
            # 预测动作，只选择最优动作
            action = agent.predict(observation)
            observation,reward,done,_ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
        return np.mean(eval_reward)




if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n
    observation_shape = env.observation_space.shape

    rpm = Memory(MEMORY_SIZE)
    model = Model(observation_shape[0],action_dim)
    algorithm = DQN(model,gamma=GAMMA,learning_rate=LEARNING_RATE)
    agent = Agent(action_dim,algorithm,e_greed=0.1,e_greed_increasement=1e-6)

    # 先往经验池里面存放一些数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env,algorithm,agent,rpm)

    max_episode = 2000

    episode = 0

    while episode < max_episode:
        for i in range(0,50):
            total_reward = run_episode(env,algorithm,agent,rpm)
            episode +=1

        eval_reward = evaluate(env,agent,render=True)
        print('apisode:{}  egreed:{}  Test reward:{}'.format(episode,agent.e_greed,eval_reward))

        env.close()







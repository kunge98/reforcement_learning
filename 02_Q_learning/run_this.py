"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # 初始化状态，环境重置
        observation = env.reset()
        while True:
            # 环境刷新
            env.render()
            # 基于观测值来挑选动作
            action = RL.choose_action(str(observation))

            # 作出动作，得到 下一个状态，奖励，有没有游戏结束
            observation_, reward, done = env.step(action)

            # 如果没有结束，则学习这一个transition
            RL.learn(str(observation), action, reward, str(observation_))

            # 更新observation
            observation = observation_

            # 跳出while循环，进入下一个循环
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # 定义环境
    env = Maze()
    # 定义学习方法Q——learning
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
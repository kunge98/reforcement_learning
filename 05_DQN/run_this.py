from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    # 首要存储一些记忆库（经验回放），一个step记录一个transition
    step = 0
    for episode in range(300):
        # 初始化状态
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # 选择动作
            action = RL.choose_action(observation)

            # 动作作为参数传入得到下一个状态，奖励，是否完成
            observation_, reward, done = env.step(action)

            # 将transition（当前状态，动作，奖励，下一状态）存储进去，重要的一步
            RL.store_transition(observation, action, reward, observation_)

            # 直到存储了200以上个transition并且以后每存储5个transition的时候就学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
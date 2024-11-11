if __name__ == '__main__':


    # Q - learning 算法实现例子

    # 探索重点的小锤子

    import pandas as pd
    import numpy as np

    # time模块是控制探索者的移动速度
    import time

    # 产生一组伪随机数
    np.random.seed(2)

    # 状态，最开始距离终点的步数
    N_STATES = 6

    # 移动的选择，向左或者向右
    ACTIONS = ['left','right']

    # 用在决策上的一种策略   e_greedy
    EPSILON = 0.9

    # learning-rate学习速率
    ALPHA = 0.1

    # 奖励的衰减值
    LAMBDA = 0.9

    # 最大回合数
    MAX_EPISODES = 13

    # 走一步花费的时间
    FRESH_TIME = 0.2

    # 建立一个Q表
    def build_Q_table(n_states,actions):
        table = pd.DataFrame(
            # 创建了一个长为n_states，宽为actions长度的全为0的数组
            np.zeros((n_states,len(actions))),
            # 将行动的值作为数组的列索引
            columns=actions
        )
        # print(table)
        return table

    # build_Q_table(N_STATES,ACTIONS)

    # 选择动作
    def choose_action(state,Q_table):

        # 获取Q表当前状态那个一行的数据赋值给state_actions
        state_actions = Q_table.iloc[state,:]

        if(np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
            # 大于epsilon的时候，从当前状态中随机选取一个
            action_name = np.random.choice(ACTIONS)
        else:
            # 获取当前状态中值最大的那个
            action_name = state_actions.argmax()
        return action_name


    # 创建环境对行为作出的反应
    def get_env_feedback(state,action):
        if action == 'right':
            if state == N_STATES - 2:
                state_ = 'terminal'
                R = 1
            else:
                state_ = state + 1
                R = 0
        else:
            R = 0
            if state == 0:
                state_ = state
            else:
                state_ = state - 1
        return state_,R


    # 更新环境,了解
    def update_env(state,episode,step_counter):
        env_list = ['-']*(N_STATES-1) + ['T']
        if state == 'terminal':
            intercation = 'Episode %s: total_steps = %s' % (episode+1,step_counter)
            print('\r{}'.format(intercation),end='')
            time.sleep(2)
            print('\r                      ',end='')
        else:
            env_list[state] = 'o'
            intercation = ''.join(env_list)
            print('\r()'.format(intercation),end='')
            time.sleep(FRESH_TIME)


    # 主循环
    def reinforcement_learning():
        Q_table = build_Q_table(N_STATES,ACTIONS)
        for episode in range(MAX_EPISODES):
            step_counter = 0
            S = 0
            # 是终止符吗
            is_terminated = False
            # 更新环境
            update_env(S,episode,step_counter)
            while not is_terminated:
                A = choose_action(S,Q_table)
                # S_为下一个状态，R为奖励
                S_,R = get_env_feedback(S,A)
                # 估计值
                q_predict = Q_table.loc[S,A]
                if S_ != 'terminal':
                    # 真实值
                    q_target = R + LAMBDA * Q_table.iloc[S_,:].max()
                else:
                    q_target = R
                    is_terminated = True

                Q_table.loc[S,A] += ALPHA * (q_target - q_predict)
                S = S_
                update_env(S,episode,step_counter+1)
                step_counter +=1
        return Q_table


    # 调用开始运行
    q_table = reinforcement_learning()
    print('\r\nQ-table:\n')
    print(q_table)


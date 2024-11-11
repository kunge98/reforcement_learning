if __name__ == '__main__':

    # 随机抽样

    from numpy.random import choice

    samples = choice(['红','黄','蓝'],size=100,p=[0.1,0.3,0.6])

    print(samples)


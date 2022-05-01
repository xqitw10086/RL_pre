"""
Solving FrozenLake environment using Policy-Iteration.

Adapted by Bolei Zhou for IERG6130. Originally from Moustafa Alzantot (malzantot@ucla.edu)
"""
# 在周老师本身的代码中使用的是env.nA与env.nS分别用于表示动作空间与状态空间
# 但由于不知道是不是因为FrozenLake-v0由于版本问题不能运行导致使用v1后这两个表示不存在
# 因此更改为对应的env.action_space.n和env.observation_space.n
# env.P[s][a]表示在状态s下执行动作a，返回prob概率，next_state下一个状态，reward奖励，done是否结束

"""问题：为什么我的结果还是很差"""

import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

RENDER=False
GAMMA=1.0

def run_episode(env, policy, gamma=GAMMA, render=RENDER):
    """ Runs an episode and return the total reward """
    """ 计算策略policy跑一个回合的奖励：输入环境、策略、衰减因子，跑一个回合返回奖励值 """
    """ 想看环境渲染将render设置为true，render默认设置为FALSE """
    obs = env.reset()  # 重置环境,因为agent每次都要到达结束再从头开始
    total_reward = 0
    step_idx = 0 # 表示折扣因子的指数，第一次是0
    while True:  # 进入循环，直接到达该次游戏结束
        if render:
            env.render()
            # 重绘环境图像
        obs, reward, done, _ = env.step(int(policy[obs]))
        # env.step完成一个时间步返回四个值：对下一刻状态，即时奖励，是否终止，调试项，描述与环境交互
        # 其输入其实是一个动作action
        total_reward += (gamma ** step_idx * reward)
        # 折扣回报
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=GAMMA, n=100):
    """ 计算策略的平均奖励 """
    scores = [run_episode(env, policy, gamma, RENDER) for _ in range(n)]
    return np.mean(scores)


""" 第一步：计算v函数，输入：环境，策略以及衰减因子，计算策略价值 """
def compute_policy_v(env, policy, gamma=GAMMA):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.observation_space.n)
    # 价值函数初始化
    eps = 1e-10  # 当精度收敛到eps时停止更新
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            # 这里相当于是确定性策略 policy_a 是一个值，不是表示概率的向量
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
            # sum函数输入对象是可迭代的内容
            # sum函数实现了对s'情况的累加，这里env.env.P[s][policy_a]可能产生多个情况，表示在当前状态使用该动作可能的下一状态可能有多个
            # 这个式子和公式其实是一样的
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v


"""  第二步：改进策略policy，通过对old_policy_v使用贪心算法，改进策略policy  """
def extract_policy(v, gamma=GAMMA):
    """ Extract the policy given a value-function """
    """ 函数中不能提供参数env """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
            # 通俗理解，对每个动作求出当前状态下使用该动作之后的价值，这个公式和上面的公式一致，此时动作已经确定了

        policy[s] = np.argmax(q_sa)
        # 选择此时的最优化策略，即结果最高的动作
    return policy


def policy_iteration(env, gamma=GAMMA):
    """ Policy-Iteration algorithm """
    """ 目标：寻找一个最终策略（不断迭代Bellman expectation backup）"""
    # policy = np.random.choice(env.env.nA, size=(env.env.nS))
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)  # 第一步
        new_policy = extract_policy(old_policy_v, gamma)  # 第二步
        if np.all(policy == new_policy):
            # 如果策略不再变化则视为已经收敛，不能提升
            print('Policy-Iteration converged at step %d.' % (i + 1))
            # i从0开始
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env_name = 'FrozenLake-v1'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma=GAMMA)
    scores = evaluate_policy(env, optimal_policy, gamma=GAMMA)
    print('Average scores = ', np.mean(scores))




# 基于周博磊老师的代码，关于冰湖问题的MDP解决，主要是策略迭代和价值迭代
import numpy as np
import gym
from gym.envs.registration import register

RENDER = False
GAMMA = 1.0


def run_episode(env, policy, gamma=GAMMA, render=RENDER):
    """ 计算policy跑一个回合的奖励，输入环境、策略、折扣因子，返回一个回合的奖励值 """
    obs = env.reset()
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()
        obs, reward, done, _ =env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx) * reward
        step_idx += 1
        if done:
            break

    return total_reward

def evaluate_policy(env, policy, gamma = GAMMA, n = 100):
    """ 计算策略的平均奖励,即利用该策略跑100次之后奖励的平均值 """
    scores = [run_episode(env, policy, gamma, RENDER) for _ in range(n)]
    return np.mean(scores)

# 策略迭代方法
def compute_policy_v(env, policy, gamma = GAMMA):
    """ 第一步：计算v函数，输入：环境，策略以及衰减因子，计算策略价值 """
    v = np.zeros(env.observation_space.n)
    eps = 1e-10
    while True:
        pre_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
           #  for p, s_, r, _ in env.env.P[s][policy_a]:
                # v[s] += p * (r + gamma * pre_v[s_])
            v[s] = sum([p * (r + gamma * pre_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if np.sum((np.fabs(pre_v - v))) <= eps:
            break
    return v

def PI_extract_policy( v, gamma = GAMMA):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            # q_sa = np.zeros(env.action_space.n)
            # for p, s_, r, _ in env.env.P[s][a]:
                # q_sa[a] += p * (r + gamma * v[s_])
           q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def policy_iteration(env, gamma = GAMMA):
    policy = np.random.choice(env.action_space.n, size = env.observation_space.n)
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = PI_extract_policy(old_policy_v, gamma)
        if np.all(policy == new_policy):
            print('策略迭代在第%d次迭代时达到收敛' %(i+1))
            break
        policy = new_policy
    return policy

# 关于价值迭代
def value_iteration(env, gamma = GAMMA):
    v = np.zeros(env.observation_space.n)
    max_iteration = 100000
    eps = 1e-20
    for i in range(max_iteration):
        pre_v = np.copy(v)
        for s in range(env.observation_space.n):
            # for a in range(env.action_space.n):
                # q_sa= sum([p * (r + gamma * pre_v(s_)) for p, s_, r, _ in env.env.P[s][a]])
            q_sa = [sum([p * (r + gamma * pre_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.action_space.n)]
            v[s] = max(q_sa)
        if(np.sum(np.fabs(pre_v - v)) <= eps):
            print('价值迭代在第%d次收敛。'%(i+1))
            break
    return v

def VI_extract_policy(v, gamma = GAMMA):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy



if __name__ == '__main__':
    env_name = 'FrozenLake-v1'
    env = gym.make(env_name)
    gamma = GAMMA

    PI_optimal_policy = policy_iteration(env, gamma)
    PI_scores = evaluate_policy(env, PI_optimal_policy, gamma)
    print('策略迭代平均成绩为', np.mean(PI_scores))

    VI_optimal_v = value_iteration(env, gamma)
    VI_optimal_policy = VI_extract_policy(VI_optimal_v, gamma)
    VI_scores = evaluate_policy(env, VI_optimal_policy, gamma, n=1000)
    print('价值迭代平均成绩为', np.mean(VI_scores))






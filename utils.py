import json
import TD3

def evaluate_policy(env, model, render, turns=3):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s)
            s_prime, r, done, info = env.step(a)

            ep_r += r
            steps += 1
            s = s_prime
            if render: env.render()

        scores += ep_r
    return scores / turns

def evaluate_policy_test(env, model, render, writer, test_time=3500):
    all_episode_reward = [0]
    for episode in range(test_time):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s)
            s_prime, r, done, info = env.step(a)

            ep_r += r
            steps += 1
            s = s_prime
            if render: env.render()
        if episode == 0:
            all_episode_reward.append(ep_r)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + ep_r * 0.1)
        writer.add_scalar('s_ep_r', all_episode_reward[-1], episode)





#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r


import numpy as np
def save_data():
    filename = './save_data/data.json'
    data = {
        's':[list(np.float64([-0.26309335, -0.13913898, -0.99117994, -0.04303949, -0.025945012, -0.22139809]))],
        'r':[11, 22]
    }
    data = list(np.arange(0,12).reshape(3,4))
    file = open(filename, 'w')
    data = json.dumps(data)
    file.write(data)
    file.close()

def load_json_data(filename, buffer, keys=['s', 'a', 'r', 'sn', 'done']):
    print("the json keys is: ", keys)
    file = open(filename, 'r')
    data = json.load(file)
    # print(len(data[keys[0]]))
    for i in range(len(data[keys[0]])):
        s = np.array(data[keys[0]][i])
        a = np.array(data[keys[1]][i])
        r = np.array(data[keys[2]][i])
        sn = np.array(data[keys[3]][i])
        done = np.array(data[keys[4]][i])
        buffer.add(s, a, r, sn, done)
    print("load json data to buffer is ok!")
    print("buffer size is ", buffer.size)
    file.close()





if __name__ == '__main__':
    filename = './save_data/export.json'
    buffer = TD3.ReplayBuffer(17, 6, int(1e5))
    load_json_data(filename, buffer)

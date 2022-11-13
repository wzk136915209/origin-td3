import numpy as np
import torch
import gym
from TD3 import TD3_Agent, ReplayBuffer, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import str2bool,Reward_adapter,evaluate_policy, evaluate_policy_test
import psutil
import mujoco_py
# import tensorflow_probability as tfb
# import tensorflow as tf
import random
from maxque import MaxQueue
import platform



'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=6, help='PV0, Lch_Cv2, Humanv2, HCv2, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=7e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

parser.add_argument('--policy_delay_freq', type=int, default=1, help='Delay frequency of Policy Updating')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--exp_noise', type=float, default=0.4, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
print(opt)

def prob_gass(mean, std, x):
    y = []
    for i in range(len(x)):
        y.append(np.exp(-np.power((x[i]-mean[i]), 2)/(2.0 * std**2))/(std*np.sqrt(2*np.pi)))
    return np.array(y)

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EnvName = ['Pendulum-v0','LunarLanderContinuous-v2','Humanoid-v2','HalfCheetah-v2','BipedalWalker-v3','BipedalWalkerHardcore-v3', 'Walker2d-v2']
    BrifEnvName = ['PV0', 'LLdV2', 'Humanv2', 'HCv2','BWv3', 'BWHv3', 'walker'] #Brief Environment Name.
    env_with_dw = [False, True, True, False, True, True, True]  # dw:die and win
    EnvIdex = opt.EnvIdex
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    expl_noise = opt.exp_noise
    max_e_steps = env._max_episode_steps
    print('Env:', EnvName[EnvIdex], '  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action,
          '  min_a:', env.action_space.low[0],'  max_e_steps:',max_e_steps )

    update_after = 2 * max_e_steps  # update actor and critic after update_after steps
    # start_steps = 10*max_e_steps #start using actor to iterate after start_steps steps
    start_steps = 0

    #Random seed config:
    random_seed = 0
    print("Random Seed: {}".format(random_seed))

    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # tf.random.set_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.set_num_threads(2)
    # torch.set_default_dtype(torch.float32)

    human_reward = 2500


    if opt.write:
        # save path
        if platform.system().lower() == 'windows':
            logdir = './data/' + EnvName[EnvIdex] + "/random" + str(random_seed) +\
                     '/human-q-15/' + str(human_reward) + "/no-clip-norm"
        elif platform.system().lower() == 'linux':
            rootpaht = "/mnt/HDD8T2/wzkfile/new/origin-td3"
            logdir = rootpaht + '/data/' + EnvName[EnvIdex] + "/random" + str(random_seed) + \
                     '/human-q-15/' + str(human_reward)  + "/clip-norm-15"
        print(logdir)
        writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists('model'):
        os.mkdir('model')

    kwargs = {
        "env_with_dw":env_with_dw[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": opt.gamma,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "policy_delay_freq": opt.policy_delay_freq,
        "writer": writer
    }


    model = TD3_Agent(**kwargs)
    print("-"*80, 'actor model')
    print(model.actor)
    print("-"*80, 'critic model')
    print(model.q_critic)
    if opt.Loadmodel:
        model.load(BrifEnvName[EnvIdex], opt.ModelIdex)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # human in the loop
    human_replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(4e5))
    human = TD3_Agent(**kwargs)
    human.load(BrifEnvName[EnvIdex],'reward=4000')
    h_s = eval_env.reset()
    h_a = human.select_action(h_s)
    human_no_action = 0
    human.human_reward = human_reward

    train_flag = True
    for i in range(10):
        model.que.push_back(0)


    if train_flag == False:
        evaluate_policy_test(env, human, opt.render, writer, 3500)
    else:

        total_steps = 0
        all_episode_reward = []
        all_episode_reward.append(0)
        episode = 0
        save_flag1 = False
        save_flag2 = False

        while episode < 3500:
            s, done, steps, r = env.reset(), False, 0, 0
            ep_r = 0
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

            if episode%10 == 0:
                writer.add_scalar('memory', mem, episode)

            '''Interact & trian'''
            while not done:
                steps += 1  #steps in one episode

                #random sample action, it will improve informace, but don't repeate again
                if total_steps < start_steps:
                    a = env.action_space.sample()
                else:
                    # a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                    #      ).clip(-max_action, max_action)  # explore: deterministic actions + noise
                    model.std = expl_noise
                    a = model.slect_action_normal(s)

                    # human in the loop q diff
                    q_diff = model.computer_q_diff(s, a)
                    maxqvalue = model.que.max_value()
                    model.que.push_back(q_diff)
                    human.human_flag = 0
                    if q_diff > maxqvalue and all_episode_reward[-1] < human.human_reward:
                        h_a = human.select_action(s)
                        if np.mean(abs(h_a - a)) > 0.2:
                            #a = h_a
                            human.human_flag = 1
                        else:
                            human_no_action += 1
                            human.human_flag = 0
                            h_a = a
                            writer.add_scalar('human-no-action', human_no_action, total_steps)


                    h_a = h_a.clip(-max_action,max_action)
                    a = a.clip(-max_action, max_action)
                    writer.add_scalar('q-diff', q_diff, total_steps)
                    if human.human_flag:
                        action = h_a
                    else:
                        action = a

                s_prime, r, done, info = env.step(action)
                r = Reward_adapter(r, EnvIdex)
                # writer.add_scalar('origin reward', r, total_steps)

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != max_e_steps):
                    dw = True  # dw: dead and win
                else:
                    dw = False

                if human.human_flag:
                    human_replay_buffer.add(s, h_a, r, s_prime, dw)
                    replay_buffer.add(s, h_a, r, s_prime, dw)
                else:
                    replay_buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                if done and total_steps > start_steps:
                    episode += 1
                    all_episode_reward.append(all_episode_reward[-1] * 0.9 + ep_r * 0.1)
                    writer.add_scalar('reward train', all_episode_reward[-1], episode)
                    writer.add_scalar('buffer human', human_replay_buffer.size, episode)
                    writer.add_scalar('buffer agent', replay_buffer.size, episode)
                    # print('EnvName:', BrifEnvName[EnvIdex], 'episode: {}'.format(episode),
                    #       'score:', all_episode_reward[-1])

                '''train if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                # if total_steps >= update_after and total_steps % opt.update_every == 0:
                #     for j in range(opt.update_every):
                if total_steps >= update_after:
                    model.train(replay_buffer, human_replay_buffer, 1, all_episode_reward[-1], human_replay_buffer.size, human.human_reward)
                    model.train(replay_buffer, human_replay_buffer, 0, all_episode_reward[-1], human_replay_buffer.size, human.human_reward)


                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    expl_noise *= opt.noise_decay
                    writer.add_scalar('expl_noise', expl_noise, global_step=total_steps)
                    score = evaluate_policy(eval_env, model, False)
                    if opt.write:
                        writer.add_scalar('reward eval', score, global_step=total_steps)

                    print('EnvName:', BrifEnvName[EnvIdex], 'steps: {}k'.format(int(total_steps/1000)), 'score:', score)
                total_steps += 1

                '''save model'''
                # if total_steps % opt.save_interval == 0:
                #     model.save(BrifEnvName[EnvIdex],total_steps)

                if episode>700 and save_flag1 and np.mean(all_episode_reward[-5:]) > 4000 and ep_r > 4000:
                    model.save(BrifEnvName[EnvIdex], 'reward=4000')
                    save_flag1 = False
                if episode>700 and save_flag2 and np.mean(all_episode_reward[-5:]) > 2500 and ep_r > 2500:
                    model.save(BrifEnvName[EnvIdex], 'reward=2500')
                    save_flag2 = False

        env.close()
        eval_env.close()



if __name__ == '__main__':
    main()





import gym
import numpy as np
np.float = float
import matplotlib.pyplot as plt
from pygifsicle import optimize
import imageio
import _pickle as pickle

from population import POPULATION
from evogym.envs import *
from evogym_wrappers import RenderWrapper, ActionSkipWrapper, RewardShapingWrapper

class MAKEGIF():

    def __init__(self, args, ind):
        self.kwargs = vars(args)
        self.ind = ind

    def run(self):
        env = gym.make(self.kwargs['task'], body=self.ind.body.body['structure'], connections=get_full_connectivity(self.ind.body.body['structure']))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = RenderWrapper(env, render_mode='img')
        if 'sparse_acting' in self.kwargs and self.kwargs['sparse_acting']:
            env = ActionSkipWrapper(env, skip=self.kwargs['act_every'])
        env = RewardShapingWrapper(env)
        env.seed(17)
        env.action_space.seed(17)
        env.observation_space.seed(17)
        env.env.env.env.env._max_episode_steps = 500

        # run the environment
        cum_reward = 0
        observation = env.reset()
        for ts in range(500):
            action = self.ind.brain.get_action(observation)
            observation, reward, done, _ = env.step(action)
            cum_reward += reward
            if type(done) == bool:
                if done:
                    break
            elif type(done) == np.ndarray:
                if done.all():
                    break
            else:
                raise ValueError('Unknown type of done', type(d))
        # print the env.imgs[0] to see what it looks like
        # don't print axis and remove white space
        #img = env.imgs[0]
        #plt.axis('off')
        #plt.imshow(img)
        #plt.savefig(self.kwargs['output_path'] + '.png', bbox_inches='tight', pad_inches=0)
        #plt.close()
        imageio.mimsave(f"{self.kwargs['output_path']}_{cum_reward}_{self.ind.body.body['name']}.gif", env.imgs, duration=(1/50.0))
        try:
            optimize(f"{self.kwargs['output_path']}_{cum_reward}_{self.ind.body.body['name']}")
        except:
            pass


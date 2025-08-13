import gym
import numpy as np
from copy import deepcopy

from evogym.envs import *
from evogym import get_full_connectivity
from evogym import EvoViewer

class RewardShapingWrapper(gym.RewardWrapper):
    """ adds a small negative reward to every step """
    def __init__(self, env):
        super(RewardShapingWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward-0.05, done, info

class RenderWrapper(gym.core.Wrapper):
    def __init__(self, env, render_mode='screen'):
        super().__init__(env)
        self.viewer = EvoViewer(env.sim, resolution=(300, 300), target_rps=120)
        self.viewer.track_objects('robot')
        self.render_mode = render_mode
        self.imgs = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.render_mode == 'screen':
            self.viewer.render(self.render_mode)
        elif self.render_mode == 'img':
            self.imgs.append(self.viewer.render(self.render_mode))
        else:
            raise NotImplementedError
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.render_mode == 'screen':
            self.viewer.render(self.render_mode)
        elif self.render_mode == 'img':
            self.imgs.append(self.viewer.render(self.render_mode))
        else:
            raise NotImplementedError
        return obs

    def close(self):
        super().close()
        self.viewer.hide_debug_window()
        self.imgs = []

class ActionSkipWrapper(gym.core.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

if __name__ == '__main__':
    from body import biped, worm
    deneme = np.array([
                [0,0,0],
                [3,1,0],
                [4,0,0]
            ])

    kwargs = {'observe_structure': True,
            'observe_voxel_volume': True,
            'observe_voxel_speed': True,
            'observe_time': False,
            'observe_time_interval': 10}

    np.float = float
    env = gym.make('Walker-v0', body=deneme, connections=get_full_connectivity(deneme))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = RenderWrapper(env, render_mode='screen')
    env = ActionSkipWrapper(env, skip=5)
    env = ActionSpaceCorrectionWrapper(env)
    #env = LocalObservationWrapper(env, **kwargs)
    #env = LocalActionWrapper(env, **kwargs)
    #env = GlobalObservationWrapper(env, **kwargs)
    #env = GlobalActionWrapper(env, **kwargs)
    env = TransformerObservationWrapper(env, **kwargs)
    env = TransformerActionWrapper(env, **kwargs)
    env = RewardShapingWrapper(env)
    env.seed(17)
    env.action_space.seed(17)
    env.observation_space.seed(17)
    print(env.env.env.env.env.env.env.env._max_episode_steps)
    env.env.env.env.env.env.env.env._max_episode_steps = 2500

    a, b = env.reset()
    print(f'a: {a}')
    cum_reward = 0
    for _ in range(2500):
        action = env.action_space.sample()
        #obs, reward, done, info = env.step(np.array([action]*10))
        #action[0] = 0.6
        #action[1] = 0.0
        #action[2] = -0.4
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        print(f'reward: {reward}, done: {done}, info: {info}')
        print(f'obs shape: {obs[0].shape}')
        print(f'action shape: {action.shape}')
        #print(f'obs: {obs}')
        #print(f'obs: {obs[0][-1]}')
        print(f'obs: {obs[0]}')
        input()
        if done:
            print(f'{_} steps')
            print(f'cumulative reward: {cum_reward}')
            exit()
    exit()



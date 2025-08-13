import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from copy import deepcopy
import _pickle as pickle
import gym
from evogym.envs import *
np.float = float

from search_space import integer_idx_to_ndarray
from evogym import get_full_connectivity
from networks import NeuralNetwork

class BRAIN(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def mutate(self):
        raise NotImplementedError

    @staticmethod
    def name(self):
        return self.name

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def is_valid(self):
        raise NotImplementedError

    def extract_brain(self):
        raise NotImplementedError

    def get_action(self, observation):
        # get the actions
        actions = self.model.forward(torch.from_numpy(observation).double())
        # turn the actions into numpy array
        actions = actions.detach().numpy()
        return actions

    def update_model(self):
        if type(self.weights) != torch.Tensor:
            self.weights = torch.from_numpy(self.weights).double()
        vector_to_parameters(self.weights, self.model.parameters())
        self.model.double()


class CENTRALIZED_RE(BRAIN):
    def __init__(self, args):
        BRAIN.__init__(self, "CENTRALIZED_RE", args)
        self.mu, self.sigma = 0, 0.1
        dummy_structure = np.ones((3,3)) * 3
        dummy_env = gym.make("Walker-v0",
                             body=dummy_structure,
                             connections=get_full_connectivity(dummy_structure))
        input_size = dummy_env.observation_space.shape[0]
        output_size = dummy_env.action_space.shape[0]

        self.model = NeuralNetwork(input_size, output_size)
        for p in self.model.parameters():
            p.requires_grad = False
        self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()

    def mutate(self):
        noise_weights = np.random.normal(self.mu, self.sigma,
                                         self.weights.shape)
        self.weights += noise_weights
        self.update_model()
        return noise_weights


class CENTRALIZED(BRAIN):
    '''
    '''
    def __init__(self, args):
        BRAIN.__init__(self, "CENTRALIZED", args)
        self.mu, self.sigma = 0, 0.1
        # input size depends on robot structure and problem
        # just create a dummy environment to get the observation and action spaces
        structure = integer_idx_to_ndarray(args.body_id, args.bounding_box) # we assume the id corresponds to a valid structure
        env = gym.make(args.task, body=structure, connections=get_full_connectivity(structure))
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]

        self.model = NeuralNetwork(input_size, output_size)
        for p in self.model.parameters():
            p.requires_grad = False
        self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()

    def mutate(self):
        noise_weights = np.random.normal(self.mu, self.sigma,
                                         self.weights.shape)
        self.weights += noise_weights
        self.update_model()
        return noise_weights

    def is_valid(self):
        return True

if __name__=="__main__":
    # Test
    ...







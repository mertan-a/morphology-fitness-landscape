import string
from copy import deepcopy
from datetime import datetime
import random
import numpy as np


class INDIVIDUAL():
    ''' individual class that contains brain and body '''

    def __init__(self, body, brain, innovation_protection_body=False, innovation_protection_brain=False):
        ''' initialize the individual class with the given body and brain

        Parameters
        ----------
        body: BODY class instance or list of BODY class instances
            defines the shape and muscle properties of the robot

        brain: BRAIN class instance
            defines the controller of the robot

        '''
        # ids
        self.parent_id = ''
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        # attributes
        self.fitness = None
        self.detailed_fitness = None
        self.age = 0
        # initialize main components
        self.body = body
        self.brain = brain

    def mutate(self):
        # handle ids
        self.parent_id = self.self_id
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        rand_int = np.random.randint(0, 100)
        nw = None
        self.self_id = 'BRAIN_' + self.self_id
        nw = self.brain.mutate()
        return nw

    def produce_offspring(self):
        '''
        produce an offspring from the current individual
        '''
        while True:
            offspring = deepcopy(self)
            offspring.mutate()
            if offspring.is_valid():
                break
        offspring.fitness = None
        return offspring

    def is_valid(self):
        '''
        check if the individual is valid
        '''
        if self.body.is_valid():
            return True
        else:
            return False

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        new = self.__class__(body=self.body, brain=self.brain)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new


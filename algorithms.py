import os
import random
import _pickle as pickle
import math
import numpy as np
import time
from copy import deepcopy
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils import get_files_in, natural_sort
from simulator import simulate_population
from make_gif import MAKEGIF
from evogym import is_connected, get_full_connectivity
from search_space import ndarray_to_integer_idx


class AFPO():
    def __init__(self, args, population):
        self.args = args
        self.best_fitness = None
        self.population = population
        # create the directory if it doesn't exist
        if not os.path.exists(self.args.rundir):
            os.makedirs(self.args.rundir)
        print('starting from scratch\n')
        self.starting_generation = 1
        self.current_generation = 0
        print('evaluating initial population\n')
        self.evaluate()
        self.population.calc_dominance()
        self.population.sort_by_objectives()
        self.best_fitness_over_time = []
        self.best_fitness_over_time.append(self.population[0].fitness)
        # find the neighbors of this morphology
        self.neighbors = self.get_neighbors()
        self.num_neighbors = len(self.neighbors)
        if self.num_neighbors == 0:
            raise ValueError('No neighbors found')
        elif self.num_neighbors < 10:
            self.transfer_to_all_neighbors = True
        else:
            self.transfer_to_all_neighbors = False
            random.shuffle(self.neighbors)
            self.chosen_neighbors = self.neighbors[:5]
            self.unchosen_neighbors = self.neighbors[5:]
        print(' transfer to neighbors')
        #self.transfer_results = {}
        #self.transfer_to_neighbors()

    def optimize(self):

        for gen in range(self.starting_generation, self.args.nr_generations):

            print('GENERATION: {}'.format(gen))
            self.do_one_generation(gen)
            self.record_keeping(gen)
        #    if gen % 1 == 0:
        #        self.transfer_to_neighbors()

        self.args.output_path = os.path.join(self.args.rundir, f'{self.args.body_id}')
        t = MAKEGIF(self.args, self.population[0])
        t.run()
        self.post_job_processing()

    def do_one_generation(self, gen):

        self.current_generation = gen
        print('PRODUCING OFFSPRINGS')
        self.produce_offsprings()
        print('EVALUATING POPULATION')
        self.evaluate()
        print('SELECTING NEW POPULATION')
        self.select()

    def produce_offsprings(self):
        '''produce offsprings from the current population
        '''
        print('population size: {}'.format(len(self.population)))
        self.population.produce_offsprings()
        print('offsprings produced')
        print('add {} random individuals'.format(self.args.nr_random_individual))
        for i in range(self.args.nr_random_individual):
            self.population.add_individual()
        print('new population size: {}\n'.format(len(self.population)))

    def evaluate(self):
        '''evaluate the current population
        '''
        # determine unevaluated individuals
        unevaluated = []
        for ind in self.population:
            if ind.fitness is None:
                unevaluated.append(ind)
        # evaluate the unevaluated individuals
        simulate_population(population=unevaluated, **vars(self.args))
        # update the fitness of the evaluated individuals
        for ind in unevaluated:
            ind.fitness = list(ind.detailed_fitness.values())[0]
        print('population evaluated\n')

    def select(self):
        '''select the individuals that will be the next generation
        '''
        self.population.calc_dominance()
        self.population.sort_by_objectives()
        # print the self_id, fitness, pareto_level, parent_id, and age of the individuals
        print('population before selection')
        for ind in self.population:
            print('self_id: {}, fitness: {}, pareto_level: {}, parent_id: {}, age: {}'.format(
                ind.self_id, ind.fitness, ind.pareto_level, ind.parent_id, ind.age))
        # choose population_size number of individuals based on pareto_level
        new_population = []
        done = False
        pareto_level = 0
        while not done:
            this_level = []
            size_left = self.population.args.nr_parents - len(new_population)
            for ind in self.population:
                if len(ind.dominated_by) == pareto_level:
                    this_level += [ind]

            # add best individuals to the new population.
            # add the best pareto levels first until it is not possible to fit them in the new_population
            if len(this_level) > 0:
                # if whole pareto level can fit, add it
                if size_left >= len(this_level):
                    new_population += this_level
                else:  # otherwise, select by sorted ranking within the level
                    new_population += [this_level[0]]
                    while len(new_population) < self.population.args.nr_parents:
                        random_num = random.random()
                        log_level_length = math.log(len(this_level))
                        for i in range(1, len(this_level)):
                            if math.log(i) / log_level_length <= random_num < math.log(i + 1) / log_level_length and \
                                    this_level[i] not in new_population:
                                new_population += [this_level[i]]
                                continue

            pareto_level += 1
            if len(new_population) == self.population.args.nr_parents:
                done = True
        self.population.individuals = new_population
        self.population.update_ages()
        # print the self_id, fitness, parent_id, and age of the individuals
        print('population after selection')
        for ind in self.population:
            print('self_id: {}, fitness: {}, parent_id: {}, age: {}'.format(
                ind.self_id, ind.fitness, ind.parent_id, ind.age))

    def record_keeping(self, gen):
        '''keeps record of important stuff'''
        self.best_fitness_over_time.append(self.population[0].fitness)

    def post_job_processing(self):
        '''post job processing'''
        print(f"processed body {self.args.body_id}")
        # save best fitness over time
        self.best_fitness_over_time = np.array(self.best_fitness_over_time)
        np.save(os.path.join(self.args.rundir, f'{self.args.body_id}_best_fitness_over_time.npy'), self.best_fitness_over_time)
        #with open(os.path.join(self.args.rundir, f'{self.args.body_id}_transfer_results.pkl'), 'wb') as f:
            #    pickle.dump(self.transfer_results, f)
        #print(f"transfer results: {self.transfer_results}")

    def get_neighbors(self):
        '''get the neighbors of the current morphology
        '''
        structure = self.population[0].body.body['structure']
        neighbors = []
        # get the neighbors
        for r in range(structure.shape[0]):
            for c in range(structure.shape[1]):
                for m in range(5):
                    if m == structure[r, c]:
                        continue
                    potential_neighbor = deepcopy(structure)
                    potential_neighbor[r, c] = m
                    if not is_connected(potential_neighbor):
                        continue
                    neighbors.append(potential_neighbor)
        return neighbors

    def transfer_to_neighbors(self):
        '''transfer the best individual to the neighbors
        '''
        to_evaluate = []
        if self.transfer_to_all_neighbors:
            for neighbor in self.neighbors:
                copy_best = deepcopy(self.population[0])
                copy_best.body.body = {"structure": neighbor, 
                                       "connections": get_full_connectivity(neighbor), 
                                       "name": ndarray_to_integer_idx(neighbor),
                                       "nr_active_voxels": np.sum( neighbor == 3 ) + np.sum( neighbor == 4 )}
                copy_best.fitness = None
                copy_best.detailed_fitness = None
                to_evaluate.append(copy_best)
        else:
            for neighbor in self.chosen_neighbors:
                copy_best = deepcopy(self.population[0])
                copy_best.body.body = {"structure": neighbor, 
                                       "connections": get_full_connectivity(neighbor), 
                                       "name": ndarray_to_integer_idx(neighbor),
                                       "nr_active_voxels": np.sum( neighbor == 3 ) + np.sum( neighbor == 4 )}
                copy_best.fitness = None
                copy_best.detailed_fitness = None
                to_evaluate.append(copy_best)
            # also choose 5 random neighbors that are not in the chosen neighbors and unique
            random_neighbors = random.sample(self.unchosen_neighbors, 5)
            for neighbor in random_neighbors:
                copy_best = deepcopy(self.population[0])
                copy_best.body.body = {"structure": neighbor, 
                                       "connections": get_full_connectivity(neighbor), 
                                       "name": ndarray_to_integer_idx(neighbor),
                                       "nr_active_voxels": np.sum( neighbor == 3 ) + np.sum( neighbor == 4 )}
                copy_best.fitness = None
                copy_best.detailed_fitness = None
                to_evaluate.append(copy_best)
        # evaluate the individuals
        simulate_population(population=to_evaluate, **vars(self.args))
        self.transfer_results[self.current_generation] = {}
        for ind in to_evaluate:
            self.transfer_results[self.current_generation][ind.body.body['name']] = ind.fitness




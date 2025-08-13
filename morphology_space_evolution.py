import os
import random
import numpy as np
import sys
import _pickle as pickle 
import string
from datetime import datetime
from copy import deepcopy
from search_space import find_neighbors_from_idx
import time

NR_PARENTS = 10
NR_RANDOM_INDIVIDUAL = 1
NR_GENERATIONS = 1000

def create_record_copy(ind):
    return deepcopy(ind)

def create_random_individual(body_idxs):
    random_individual = {}
    random_individual['uid'] = ''.join(random.sample(string.ascii_uppercase, k=5)) + '_' + datetime.now().strftime('%H%M%S%f')
    random_individual['lineage'] = []
    random_individual['fitness'] = None
    random_individual['age'] = 0
    random_individual['body_idx'] = random.choice(body_idxs)
    return random_individual
    
def evaluate_population(population, morphology_space):
    for individual in population:
        # get the body index
        body_idx = individual['body_idx']
        # get the fitness
        fitness = morphology_space[body_idx]
        # assign the fitness to the individual
        individual['fitness'] = fitness
    return population

def create_offspring(parent, morphology_space):
    # create the offspring
    offspring = deepcopy(parent)
    # update lineage
    offspring['lineage'].append(parent['uid'])
    # new uid
    offspring['uid'] = ''.join(random.sample(string.ascii_uppercase, k=5)) + '_' + datetime.now().strftime('%H%M%S%f')
    # reset fitness
    offspring['fitness'] = None
    # find one step neighbors
    neighbors = find_neighbors_from_idx(parent['body_idx'])
    # choose a random neighbor to be the new body
    offspring['body_idx'] = random.choice(neighbors)
    # update the uid
    offspring['uid'] = 'BODY_' + offspring['uid']
    return offspring

def run_morphology_space_evolution_afpo(morphology_space):

    ##########################
    # initialization
    ##########################
    # list of individuals, making up the population
    population = []
    for i in range(NR_PARENTS):
        random_individual = create_random_individual(list(morphology_space.keys()))
        population.append(random_individual)
    # evaluate the initial population
    population = evaluate_population(population, morphology_space)
    # to keep record
    record = {}
    record['best_fitness_over_time'] = [] # list of scalar values
    record['best_individual'] = None # dictionary containing the best individual
    record['populations'] = [] # list of tuples, first element is the parent population, second element is the offspring population

    ##########################
    # optimization
    ##########################
    for gen in range(NR_GENERATIONS):
        print(f"GENERATION {gen}")

        #### produce offsprings
        offsprings = []
        for parent in population:
            # age the parent
            parent['age'] += 1
            # create an offspring
            offspring = create_offspring(parent, morphology_space)
            offsprings.append(offspring)
        # add random individuals
        for i in range(NR_RANDOM_INDIVIDUAL):
            random_individual = create_random_individual(list(morphology_space.keys()))
            offsprings.append(random_individual)

        #### evaluate the offsprings
        evaluate_population(offsprings, morphology_space)

        #### record keeping
        parent_population = [ create_record_copy(parent) for parent in population ]
        offspring_population = [ create_record_copy(offspring) for offspring in offsprings ]
        record['populations'].append((parent_population, offspring_population))

        #### select new population
        pool = population + offsprings
        # calculate dominance for each individual
        for i, individual in enumerate(pool):
            individual['dominated_by_counter'] = 0
            for j, other_individual in enumerate(pool):
                if i == j:
                    continue
                if individual['fitness'] < other_individual['fitness'] and individual['age'] >= other_individual['age']:
                    individual['dominated_by_counter'] += 1
        # sort by dominance (when equal, by fitness)
        pool.sort(key=lambda x: x['fitness'], reverse=True)
        pool.sort(key=lambda x: x['dominated_by_counter'])
        # select the parents
        population = pool[:NR_PARENTS]

        #### record keeping
        record['best_fitness_over_time'].append(max([ind['fitness'] for ind in population]))
        record['best_individual'] = max(population, key=lambda x: x['fitness'])

    return record

if __name__ == '__main__':
    # read updated results 
    updated_results_w_long = np.load('results/updated_results_w_long.pkl', allow_pickle=True)
    raw_results = np.load('results/raw_results.npy', allow_pickle=True).item()
    raw_results = {k: v[-1] for k, v in raw_results.items()}
    # 
    records = []
    for i in range(100):
        print(f'Running iteration {i}')
        rec = run_morphology_space_evolution_afpo(raw_results)
        records.append(rec)

    with open('results/morphology_space_evolution_afpo_vanilla.pkl', 'wb') as f:
        pickle.dump(records, f)
    

    #records = []
    #for i in range(100):
    #    print(f'Running iteration {i}')
    #    rec = run_morphology_space_evolution_afpo(updated_results_w_long)
    #    records.append(rec)

    #with open('results/morphology_space_evolution_afpo.pkl', 'wb') as f:
    #    pickle.dump(records, f)


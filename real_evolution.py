'''
this script 
- uses argparse to determine the jobs to run 
    a either runs the job 
    b1 writes the .sh file for each job
    b2 submits the jobs
    b3 deletes the .sh files
'''
import os
import time
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser(description='run jobs')

# experiment related arguments
parser.add_argument('-ros', '--run_or_slurm', type=str,
                    help='run the job directly or submit it to slurm', choices=['slurm', 'run'])
parser.add_argument('-sq', '--slurm_queue', type=str,
                    help='choose the job queue for the slurm submissions', choices=['short', 'week', 'bluemoon'])
parser.add_argument('-sp', '--saving_path', help='path to save the experiment')
parser.add_argument('-n', '--n_jobs', type=int, default=1,
                    help='number of jobs to submit')
# evolutionary algorithm related arguments
parser.add_argument('--evolutionary_algorithm', '-ea', type=str,
                    choices=['afpo', 'qd'], help='choose the evolutionary algorithm')
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')

parser.add_argument('-r', '--repetition', type=int,
                    help='repetition number, dont specify this if you want it to be determined automatically', nargs='+')
# testing
parser.add_argument('--output_path', '-op', help='path to the gif file')

# internal use only
parser.add_argument('--local', '-l', action='store_true')

args = parser.parse_args()

def run(args):
    import random
    import numpy as np
    import multiprocessing
    import torch
    import _pickle as pickle

    from utils import prepare_rundir

    multiprocessing.set_start_method('spawn')

    # run the job directly
    if args.repetition is None:
        args.repetition = [1]
    rundir = prepare_rundir(args)
    args.rundir = rundir
    print('rundir', rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING'):
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.repetition[0]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # run evolution
    if not os.path.exists(args.rundir):
        os.makedirs(args.rundir)
    with open(args.rundir + '/RUNNING', 'w') as f:
        pass
    if args.evolutionary_algorithm == 'afpo':
        record = run_afpo(args)
    elif args.evolutionary_algorithm == 'qd':
        record = run_qd(args)
    else:
        raise ValueError('evolutionary_algorithm must be either afpo or qd')

    # save the record
    with open(args.rundir + '/record.pkl', 'wb') as f:
        pickle.dump(record, f)

    # delete running file
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # write a file to indicate that the job finished successfully
    with open(args.rundir + '/FINISHED', 'w') as f:
        pass

def submit_slurm(args, resubmit=False):
    # submit the job to slurm
    base_string = '#!/bin/sh\n\n'
    base_string += '#SBATCH --partition=' + args.slurm_queue + ' # Specify a partition \n\n'
    base_string += '#SBATCH --nodes=1  # Request nodes \n\n'
    if args.nr_parents < 50:
        base_string += '#SBATCH --ntasks=' + str(args.nr_parents*2) + '  # Request some processor cores \n\n'
    else:
        base_string += '#SBATCH --ntasks=100 # Request some processor cores \n\n'
    base_string += '#SBATCH --job-name=eg_ex_re  # Name job \n\n'
    base_string += '#SBATCH --output=outs/%x_%j.out  # Name output file \n\n'
    base_string += '#SBATCH --mail-user=alican.mertan@uvm.edu  # Set email address (for user with email "usr1234@uvm.edu") \n\n'
    base_string += '#SBATCH --mail-type=FAIL   # Request email to be sent at begin and end, and if fails \n\n'
    base_string += '#SBATCH --mem-per-cpu=10GB  # Request 16 GB of memory per core \n\n'

    base_string += 'cd /users/a/m/amertan/workspace/EVOGYM/evogym-exhaustive/ \n'
    base_string += 'spack load singularity@3.7.1\n'
    base_string += 'singularity exec --bind ../../../scratch/evogym_experiments:/scratch_evogym_experiments evogym_numba.sif xvfb-run -a python3 '

    # for each job
    for i in range(args.n_jobs):
        # create a string to write to the .sh file
        string_to_write = base_string
        string_to_write += 'real_evolution.py '
        # iterate over all of the arguments
        dict_args = deepcopy(vars(args))
        # handle certain arguments differently
        if 'run_or_slurm' in dict_args:
            dict_args['run_or_slurm'] = 'run'
        if 'n_jobs' in dict_args:
            dict_args['n_jobs'] = 1
        if 'repetition' in dict_args:
            if dict_args['repetition'] is None:
                dict_args['repetition'] = i+1
            else:
                dict_args['repetition'] = dict_args['repetition'][i]
        if 'rundir' in dict_args: # rundir might be in, delete it
            del dict_args['rundir']
        # write the arguments
        for key in dict_args:
            # key can be None, skip it in that case
            if dict_args[key] is None:
                continue
            # if the key is a list, we need to iterate over the list
            if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
                string_to_write += '--' + key + ' '
                for item in dict_args[key]:
                    string_to_write += str(item) + ' '
            elif isinstance(dict_args[key], bool):
                if dict_args[key]:
                    string_to_write += '--' + key + ' '
            else:
                string_to_write += '--' + key + ' ' + str(dict_args[key]) + ' '
        # job submission or resubmission
        if resubmit == False: # this process can call sbatch since it is not in container
            # write to the file
            with open('job.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')
            # submit the job
            os.system('sbatch job.sh')
            # sleep for a second
            time.sleep(0.1)
            # remove the job file
            os.remove('job.sh')
            # sleep for a second
            time.sleep(0.1)
        else: # this process is in container, so it cannot call sbatch. there is shell script running that will check for sh files and submit them
            import random
            import string
            # write to the file
            with open('resubmit_'+''.join(random.choices(string.ascii_lowercase, k=5))+'.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')

def run_afpo(args):
    import _pickle as pickle

    ##########################
    # initialization
    ##########################
    # list of body indices
    with open('results/feasible_body_idxs.pkl', 'rb') as f:
        feasible_body_idxs = pickle.load(f)
    # list of individuals, making up the population
    population = []
    for i in range(args.nr_parents):
        random_individual = create_random_individual(args, feasible_body_idxs)
        population.append(random_individual)
    # evaluate the initial population
    simulate_population(population)
    # to keep record
    record = {}
    record['best_fitness_over_time'] = [] # list of scalar values
    record['best_individual'] = None # dictionary containing the best individual
    record['populations'] = [] # list of tuples, first element is the parent population, second element is the offspring population


    ##########################
    # optimization
    ##########################
    for gen in range(args.nr_generations):
        print(f"GENERATION {gen}\n")

        #### produce offsprings
        offsprings = []
        for parent in population:
            # age the parent
            parent['age'] += 1
            # create an offspring
            offspring = create_offspring(parent, feasible_body_idxs)
            offsprings.append(offspring)
        # add random individuals
        for i in range(args.nr_random_individual):
            random_individual = create_random_individual(args, feasible_body_idxs)
            offsprings.append(random_individual)

        #### evaluate the offsprings
        simulate_population(offsprings)

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
        population = pool[:args.nr_parents]

        #### record keeping
        record['best_fitness_over_time'].append(max([ind['fitness'] for ind in population]))
        record['best_individual'] = max(population, key=lambda x: x['fitness'])

    return record

def run_qd(args):
    import _pickle as pickle

    ##########################
    # initialization
    ##########################
    # list of body indices
    with open('results/paper/feasible_body_idxs.pkl', 'rb') as f:
        feasible_body_idxs = pickle.load(f)
    # archive of individuals
    archive = {}
    for i in range(args.nr_parents):
        random_individual = create_random_individual(args, feasible_body_idxs)
        ri_bin = determine_bin_index(random_individual)
        archive[ri_bin] = random_individual
    # evaluate the initial archive
    simulate_population(list(archive.values()))
    # to keep record
    record = {}
    record['best_fitness_over_time'] = [] # list of scalar values
    record['best_individual'] = None # dictionary containing the best individual
    record['populations'] = [] # list of populations, first element is the parent population, second element is the offspring population


    ##########################
    # optimization
    ##########################
    for gen in range(args.nr_generations):
        print(f"GENERATION {gen}\n")

        #### produce offsprings
        offsprings = []
        for parent in archive.values():
            # age the parent
            parent['age'] += 1
            # create an offspring
            offspring = create_offspring(parent, feasible_body_idxs)
            offsprings.append(offspring)
        # add random individuals
        for i in range(args.nr_random_individual):
            random_individual = create_random_individual(args, feasible_body_idxs)
            offsprings.append(random_individual)

        #### evaluate the offsprings
        simulate_population(offsprings)

        #### record keeping
        parent_population = [ create_record_copy(parent) for parent in archive.values() ]
        offspring_population = [ create_record_copy(offspring) for offspring in offsprings ]
        record['populations'].append((parent_population, offspring_population))

        #### select new population
        for os in offsprings:
            os_bin = determine_bin_index(os)
            if os_bin not in archive:
                archive[os_bin] = os
            elif os['fitness'] > archive[os_bin]['fitness']:
                archive[os_bin] = os

        #### record keeping
        record['best_fitness_over_time'].append(max([ind['fitness'] for ind in archive.values()]))
        record['best_individual'] = max(archive.values(), key=lambda x: x['fitness'])

    return record
    
def determine_bin_index(ind):
    import numpy as np
    # get the body structure 
    body_structure = ind['body_structure']
    # count number of active voxels (3s and 4s)
    num_active_voxels = np.sum(body_structure.flatten() > 2)
    # count number of passive voxels (1s and 2s, but not 0s)
    num_passive_voxels = np.sum((body_structure.flatten() > 0) & (body_structure.flatten() < 3))
    # determine the bin index
    return (num_active_voxels, num_passive_voxels)


def create_random_individual(args, feasible_body_idxs):
    import random
    import string
    from datetime import datetime
    from search_space import ndarray_to_integer_idx, integer_idx_to_ndarray, find_neighbors_from_structure, find_neighbors_from_idx
    from brain import CENTRALIZED_RE

    random_individual = {}
    random_individual['uid'] = ''.join(random.sample(string.ascii_uppercase, k=5)) + '_' + datetime.now().strftime('%H%M%S%f')
    random_individual['lineage'] = []
    random_individual['fitness'] = None
    random_individual['age'] = 0
    random_individual['body_idx'] = random.choice(feasible_body_idxs)
    random_individual['body_structure'] = integer_idx_to_ndarray(random_individual['body_idx'], (3,3))
    random_individual['brain'] = CENTRALIZED_RE(args)

    return random_individual

def create_record_copy(ind):
    return {k: v for k, v in ind.items() if k not in ['brain', 'body_structure']}

def create_offspring(parent, feasible_body_idxs):
    import random
    import string
    from datetime import datetime
    import numpy as np
    from copy import deepcopy
    from search_space import ndarray_to_integer_idx, integer_idx_to_ndarray, find_neighbors_from_structure, find_neighbors_from_idx

    # create the offspring
    offspring = deepcopy(parent)
    # update lineage
    offspring['lineage'].append(parent['uid'])
    # new uid
    offspring['uid'] = ''.join(random.sample(string.ascii_uppercase, k=5)) + '_' + datetime.now().strftime('%H%M%S%f')
    # reset fitness
    offspring['fitness'] = None
    # 50% chance of body mutation
    if random.random() < 0.5:
        # find one step neighbors
        neighbors = find_neighbors_from_idx(parent['body_idx'])
        # choose a random neighbor to be the new body
        offspring['body_idx'] = random.choice(neighbors)
        # update the body structure
        offspring['body_structure'] = integer_idx_to_ndarray(offspring['body_idx'], (3,3))
        # update the uid
        offspring['uid'] = 'BODY_' + offspring['uid']
    else:
        # brain mutation
        offspring['brain'].mutate()
        offspring['uid'] = 'BRAIN_' + offspring['uid']
        assert not np.all(np.array(offspring['brain'].weights == parent['brain'].weights)), f"{offspring['brain'].weights} \n {parent['brain'].weights}"


    return offspring
    
def simulate_population(population):
    import multiprocessing

    # run the simulation
    finished = False
    while not finished:
        with multiprocessing.Pool(processes=len(population)) as pool:
            results_f = pool.map_async(simulate_ind, population)
            try:
                results = results_f.get(timeout=580)
                finished = True
            except multiprocessing.TimeoutError:
                print('TimeoutError')
                pass
    # assign fitness
    for r in results:
        uid, cum_reward = r
        for i in population:
            if i['uid'] == uid:
                i['fitness'] = cum_reward

def simulate_ind(ind):
    import gym 
    import numpy as np

    from evogym_wrappers import ActionSkipWrapper, RewardShapingWrapper
    from evogym import get_full_connectivity

    # check if the individual has fitness already assigned (e.g. from previous subprocess run. sometimes process hangs and does not return, all the population is re-submitted to the queue)
    if ind['fitness'] is not None:
        return ind['uid'], ind['fitness']
    # get the env
    env = gym.make("Walker-v0", 
                   body=ind['body_structure'], 
                   connections=get_full_connectivity(ind['body_structure']))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ActionSkipWrapper(env, skip=5)
    env = RewardShapingWrapper(env)
    env.seed(17)
    env.action_space.seed(17)
    env.observation_space.seed(17)
    env.env.env.env.env._max_episode_steps = 500
    # record keeping
    cum_reward = 0
    # run simulation
    obs = env.reset()
    for t in range(500):
        # pad observations if it is shorter than 34
        if len(obs) < 34:
            obs = np.pad(obs, (0, 34-len(obs)))
        # collect actions
        actions = ind['brain'].get_action(obs)
        # filter actions based on robot structure
        filter = ind['body_structure'].flatten() > 2
        actions = actions[filter]
        # step
        obs, r, d, i = env.step(actions)
        # record keeping
        cum_reward += r
        # break if done
        if d:
            break
    return ind['uid'], cum_reward


if __name__ == '__main__':

    if args.run_or_slurm == 'run':
        # run the job
        run(args)
    elif args.run_or_slurm == 'slurm':
        # submit the job
        submit_slurm(args)
    else:
        raise ValueError('run_or_slurm must be either run or slurm')


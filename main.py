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

# sim
parser.add_argument('--sim_sif', '-ss', type=str, default='evogym_numba.sif', 
                    choices=['evogym_numba.sif'], help='simulation sif')
# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['Walker-v0'], default='Walker-v0')
# experiment related arguments
parser.add_argument('-ros', '--run_or_slurm', type=str,
                    help='run the job directly or submit it to slurm', choices=['slurm', 'run'])
parser.add_argument('-sq', '--slurm_queue', type=str,
                    help='choose the job queue for the slurm submissions', choices=['short', 'week'])
parser.add_argument('-cpu', '--cpu', type=int, 
                    help='number of cpu cores requested for slurm job')
parser.add_argument('-sp', '--saving_path', help='path to save the experiment')
# evolutionary algorithm related arguments
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')
# softrobot related arguments
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int,
                    help='Bounding box dimensions (x,y). e.g.IND_SIZE=(6, 6)->workspace is a rectangle of 6x6 voxels') # trying to get rid of this
# controller
parser.add_argument('--observe_time', '-ot', action='store_true')
parser.add_argument('--observe_time_interval', '-oti', type=int)
parser.add_argument('--sparse_acting', '-sa', action='store_true')
parser.add_argument('--act_every', '-ae', type=int)

# internal use only
parser.add_argument('--local', '-l', action='store_true')

args = parser.parse_args()

def run(args):
    import random
    import numpy as np
    import multiprocessing
    import time

    from utils import prepare_rundir, ATOMICOPEN
    from population import POPULATION
    from algorithms import AFPO
    import torch

    multiprocessing.set_start_method('spawn', force=True)

    # get the next unprocessed id
    file_name = f"search_space_({args.bounding_box[0]}, {args.bounding_box[1]}).txt"
    # first check the if file exists
    if not os.path.isfile(file_name):
        raise ValueError(f"search space file {file_name} does not exist")
    # if it exists, read the file, and get the next unprocessed id, and write the file back (with the next unprocessed id removed)
    if not os.path.exists(file_name):
        raise ValueError(f"search space file {file_name} does not exist")

    with ATOMICOPEN(file_name, 'r+') as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise ValueError(f"search space file {file_name} is empty")
        next_unprocessed_id = int(lines[0])
        lines = lines[1:]
        f.seek(0)
        f.writelines(lines)
        f.truncate()
    args.body_id = next_unprocessed_id

    # run the job directly
    if hasattr(args, 'rundir'):
        ...
    else:
        rundir = prepare_rundir(args)
        args.rundir = rundir
    print('rundir', args.rundir)

    # Initializing the random number generator for reproducibility
    SEED = next_unprocessed_id
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # create population
    population = POPULATION(args=args)

    # Setting up the optimization algorithm and runnning
    pareto_optimization = AFPO(args=args, population=population)
    pareto_optimization.optimize()

    # now the id is processed, we can write it to the processed file
    with ATOMICOPEN(f"processed_{file_name}", 'a+') as f:
        f.write(str(next_unprocessed_id) + '\n')
        file_names = os.listdir(args.rundir)
        file_names = [os.path.join(args.rundir, f) for f in file_names if f.endswith('.npy') or f.endswith('.gif') or f.endswith('.pkl')]
        if len(file_names)>1000:
            # zip all the files with .npy and .gif and .pkl extensions
            import zipfile
            random_saving_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
            with zipfile.ZipFile(f'{args.rundir}/{random_saving_name}.zip', 'w') as zipf:
                for f in file_names:
                    print(f)
                    zipf.write(f)
            # remove the files
            for f in file_names:
                os.remove(f)

    if not args.local:
        if hasattr(args, 'output_path') and args.output_path is not None:
            args.output_path = None
        if hasattr(args, 'body_id') and args.body_id is not None:
            args.body_id = None
        submit_slurm(args)
    else:
        run(args)

def submit_slurm(args):
    # submit the job to slurm
    base_string = '#!/bin/sh\n\n'
    base_string += '#SBATCH --partition=' + args.slurm_queue + ' # Specify a partition \n\n'
    base_string += '#SBATCH --nodes=1  # Request nodes \n\n'
    if args.cpu is None:
        if args.nr_parents < 50:
            base_string += '#SBATCH --ntasks=' + str(args.nr_parents*2) + '  # Request some processor cores \n\n'
        else:
            base_string += '#SBATCH --ntasks=100 # Request some processor cores \n\n'
    else:
        base_string += '#SBATCH --ntasks=' + str(args.cpu) + '  # Request some processor cores \n\n'
    base_string += '#SBATCH --job-name=search_space  # Name job \n\n'
    base_string += '#SBATCH --output=outs/%x_%j.out  # Name output file \n\n'
    base_string += '#SBATCH --mail-user=alican.mertan@uvm.edu  # Set email address (for user with email "usr1234@uvm.edu") \n\n'
    base_string += '#SBATCH --mail-type=FAIL   # Request email to be sent at begin and end, and if fails \n\n'
    base_string += '#SBATCH --mem-per-cpu=10GB  # Request x GB of memory per core \n\n'
    base_string += '#SBATCH --time=0-02:00:00  # Request 0 hours and 30 minutes of runtime \n\n'

    base_string += 'cd /users/a/m/amertan/workspace/EVOGYM/evogym-exhaustive/ \n'
    base_string += 'spack load singularity@3.7.1\n'
    base_string += 'trap \'kill -INT "${PID}"; wait "${PID}"; handler\' USR1 SIGINT SIGTERM \n'
    base_string += 'singularity exec --bind ../../../scratch/evogym_experiments:/scratch_evogym_experiments ' + args.sim_sif + ' xvfb-run -a python3 '

    # create a string to write to the .sh file
    string_to_write = base_string
    string_to_write += 'main.py '
    # iterate over all of the arguments
    dict_args = deepcopy(vars(args))
    # handle certain arguments differently
    if 'run_or_slurm' in dict_args:
        dict_args['run_or_slurm'] = 'run'
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
    # job submission
    import random
    import string
    # write to the file
    with open('resubmit_'+''.join(random.choices(string.ascii_lowercase, k=5))+'.sh', 'w') as f:
        f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')


if __name__ == '__main__':

    if args.run_or_slurm == 'run':
        # run the job
        run(args)
    elif args.run_or_slurm == 'slurm':
        # submit the job
        submit_slurm(args)
    else:
        raise ValueError('run_or_slurm must be either run or slurm')






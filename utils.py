import numpy as np
import os
import shutil
from copy import deepcopy
import inspect
import random
import re
import xml.etree.ElementTree as ET

# filesystem related
def prepare_rundir(args):

    # decide where the experiment will be saved
    if args.saving_path is None:
        run_dir = "experiments/"
    else:
        # if saving path starts with '/', then it is an absolute path
        if args.saving_path[0] == '/':
            args.saving_path = os.path.relpath(args.saving_path)
        run_dir = args.saving_path
        # make sure there is a trailing slash
        if not run_dir.endswith("/"):
            run_dir += "/"
    dict_args = deepcopy(vars(args))

    # remove the arguments that are not relevant for the run_dir
    if 'run_or_slurm' in dict_args:
        dict_args.pop('run_or_slurm')
    if 'slurm_queue' in dict_args:
        dict_args.pop('slurm_queue')
    if 'cpu' in dict_args:
        dict_args.pop('cpu')
    if 'saving_path' in dict_args:
        dict_args.pop('saving_path')
    if 'local' in dict_args:
        dict_args.pop('local')
    if 'body_id' in dict_args:
        dict_args.pop('body_id')
    if 'n_jobs' in dict_args:
        dict_args.pop('n_jobs')

    # write the arguments
    for key_counter, key in enumerate(dict_args):
        # key can be None, skip it in that case
        if dict_args[key] is None:
            continue
        to_add = ''
        key_string = ''
        for k in key.split('_'):
            key_string += k[0]
        # if the key is a list, we need to iterate over the list
        if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
            to_add += '.' + key_string
            for i, item in enumerate(dict_args[key]):
                to_add += '-' + str(item)
        elif isinstance(dict_args[key], bool):
            if dict_args[key]:
                to_add += '.' + key_string + '-True'
        elif "/" in str(dict_args[key]):
            processed_string = str(dict_args[key])
            processed_string = processed_string.split('/')[-1]
            processed_string = processed_string.split('.')[0]
            to_add += '.' + key_string + '-' + processed_string
        else:
            to_add = '.' + key_string + '-' + str(dict_args[key])

        if run_dir.endswith('/') and to_add.startswith('.'):
            run_dir += to_add[1:]
        else:
            run_dir += to_add
            
    return run_dir

def get_immediate_subdirectories_of(directory):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all subdirectories
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def get_files_in(directory, extension=None):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all files
    if extension is None:
        return [name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name))]
    else:
        return [name for name in os.listdir(directory)
                if (os.path.isfile(os.path.join(directory, name)) and
                    os.path.splitext(name)[1] == extension)]

# pareto front related
def natural_sort(l, reverse):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


# file locking for concurrent access to the same file
# the solution is taken from https://stackoverflow.com/questions/489861/locking-a-file-in-python
import fcntl
def lock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
def unlock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)
# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
###### This file opener *must* be used in a "with" block.
class ATOMICOPEN:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, path, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file = open(path,*args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):        
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        unlock_file(self.file)
        self.file.close()
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if (exc_type != None): return False
        else:                  return True   





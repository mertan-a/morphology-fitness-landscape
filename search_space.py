import sys
import os
#import gym
import numpy as np

#from evogym.envs import *
from evogym import is_connected


def from_base_five_to_ndarray(base_five_str, bounding_box):
    ''' turn a base five string into a numpy array '''
    # turn the string into a list of integers
    base_five_list = [int(i) for i in base_five_str]
    # turn the list into a numpy array
    integer_array = np.array(base_five_list)
    # reshape the array to the bounding box
    integer_array = integer_array.reshape(bounding_box)
    return integer_array

def ndarray_to_integer_idx(ndarray):
    """ turn a numpy array into an integer index """
    # turn the numpy array into a base five string
    base_five_str = ''.join(ndarray.flatten().astype(str))
    # turn the base five string into an integer
    idx = int(base_five_str, 5)
    return idx

def integer_idx_to_ndarray(idx, bounding_box):
    """ turn an integer index into a numpy array """
    # convert the integer to binary
    base_five_str = np.base_repr(idx, base=5)
    # check if the base five string is the same length as the bounding box
    if len(base_five_str) < bounding_box[0]*bounding_box[1]:
        # pad the base five string with zeros
        base_five_str = '0'*(bounding_box[0]*bounding_box[1] - len(base_five_str)) + base_five_str
    # turn the base five string into a numpy array
    integer_array = from_base_five_to_ndarray(base_five_str, bounding_box)
    #print(f"base 10 value: {idx}, base 5 value: {base_five_str}, back to base 10: {ndarray_to_integer_idx(integer_array)}")
    return integer_array

def create_seach_space(bounding_box):
    print(f"bounding box: {bounding_box}")
    num_of_configs = 5**(bounding_box[0]*bounding_box[1])
    print(f"total number of possible configurations: {num_of_configs}")
    # open a txt file to write the results
    with open(f'search_space_{bounding_box}.txt', 'w') as f:
        for i in range(num_of_configs):
            if i % 10000 == 0:
                print(f"progress: {float(i)*100/num_of_configs} in total: {i} out of {num_of_configs} configurations checked.")
            robot_structure = integer_idx_to_ndarray(i, bounding_box)
            # make some controls
            if not is_connected(robot_structure):
                continue
            if np.sum(robot_structure > 0) < 3:
                continue
            if np.sum(robot_structure == 3) + np.sum(robot_structure == 4) < 3:
                continue
            # write the i to the file
            f.write(str(i) + '\n')

def find_neighbors_from_structure(robot_structure):
    """ given a robot structure, find the neighbors 

    Parameters
    ----------
    robot_structure : np.ndarray
        the robot structure

    Returns
    -------
    neighbors : list
        a list of neighbors, where each neighbor is a np.ndarray
    """
    neighbors = []
    for r in range(robot_structure.shape[0]):
        for c in range(robot_structure.shape[1]):
            for m in range(5):
                if m == robot_structure[r, c]:
                    continue
                potential_neighbor = robot_structure.copy()
                potential_neighbor[r, c] = m
                if not is_connected(potential_neighbor):
                    continue
                if np.sum(potential_neighbor > 0) < 3:
                    continue
                if np.sum(potential_neighbor == 3) + np.sum(potential_neighbor == 4) < 3:
                    continue
                neighbors.append(potential_neighbor)
    return neighbors

def find_neighbors_from_idx(idx, bounding_box=(3,3)):
    ''' given an integer index, find the neighbors

    Parameters
    ----------
    idx : int
        the integer index
    bounding_box : tuple
        the bounding box

    Returns
    -------
    neighbors : list
        a list of neighbors, where each neighbor is an integer index
    '''
    robot_structure = integer_idx_to_ndarray(idx, bounding_box)
    neighbors = find_neighbors_from_structure(robot_structure)
    neighbors_idx = [ndarray_to_integer_idx(neighbor) for neighbor in neighbors]
    return neighbors_idx

if __name__ == '__main__':
    # from system arguments, determine whether we are testing or creating the search space
    if sys.argv[1] == 'create':
        # get the bounding box
        bounding_box = (int(sys.argv[2]), int(sys.argv[3]))
        # create the search space
        create_seach_space(bounding_box)
    elif sys.argv[1] == 'test':
        # get the integer index
        idx = int(sys.argv[2])
        # get the bounding box
        bounding_box = (int(sys.argv[3]), int(sys.argv[4]))
        # turn the integer index into a numpy array
        robot_structure = integer_idx_to_ndarray(idx, bounding_box)
        print(robot_structure)
    else:
        print('Invalid argument')
        sys.exit(1)


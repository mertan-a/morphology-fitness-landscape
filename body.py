import numpy as np

from evogym import get_full_connectivity
from search_space import integer_idx_to_ndarray

class BODY(object):
    def __init__(self, type):
        self.type = type
        self.nr_active_voxels = 0

    def mutate(self):
        raise NotImplementedError

    def to_phenotype(self):
        raise NotImplementedError

    def is_valid(self):
        raise NotImplementedError

    def count_active_voxels(self, structure):
        hs = np.sum( structure == 3 )
        vs = np.sum( structure == 4 )
        return hs + vs

    def count_existing_voxels(self, structure):
        return np.sum( structure != 0 )

class FIXED_BODY(BODY):
    def __init__(self, args):
        BODY.__init__(self, "fixed")
        structure = integer_idx_to_ndarray(args.body_id, args.bounding_box) # we assume the id corresponds to a valid structure
        connections = get_full_connectivity(structure)
        self.body = {"structure": structure, "connections": connections, "name": args.body_id, "nr_active_voxels": self.count_active_voxels(structure)}
        self.nr_active_voxels += self.count_active_voxels(structure)

    def mutate(self):
        raise TypeError("Cannot mutate fixed body")

    def is_valid(self):
        return True

if __name__ == '__main__':
    # Test
    from argparse import Namespace
    args = Namespace(body_id=5, bounding_box=(3, 3))
    fixed_body = FIXED_BODY(args)
    print(fixed_body.body["structure"])


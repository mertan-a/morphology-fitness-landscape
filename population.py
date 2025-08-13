import operator
import numpy as np

from individual import INDIVIDUAL
from body import FIXED_BODY
from brain import CENTRALIZED


class POPULATION(object):
    """A population of individuals"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.individuals = []
        self.non_dominated_size = 0

        while len(self) < self.args.nr_parents:
            self.add_individual()

    def add_individual(self):
        valid = False
        while not valid:
            # body
            body = FIXED_BODY(args=self.args)
            # brain
            brain = CENTRALIZED(args=self.args)
            # individual
            ind = INDIVIDUAL(body=body, brain=brain)
            if ind.is_valid():
                self.individuals.append(ind)
                valid = True

    def produce_offsprings(self):
        """Produce offspring from the current population."""
        offspring = []
        for counter, ind in enumerate(self.individuals):
            offspring.append(ind.produce_offspring())
        self.individuals.extend(offspring)

    def calc_dominance(self):
        """Determine which other individuals in the population dominate each individual."""

        # if tied on all objectives, give preference to newer individual
        self.sort(key="age", reverse=False)

        # clear old calculations of dominance
        self.non_dominated_size = 0
        for ind in self:
            ind.dominated_by = []
            ind.pareto_level = 0

        for ind in self:
            for other_ind in self:
                if other_ind.self_id != ind.self_id:
                    if self.dominated_in_multiple_objectives(ind, other_ind) and (ind.self_id not in other_ind.dominated_by):
                        ind.dominated_by += [other_ind.self_id]

            ind.pareto_level = len(ind.dominated_by)  # update the pareto level

            # update the count of non_dominated individuals
            if ind.pareto_level == 0:
                self.non_dominated_size += 1

    def dominated_in_multiple_objectives(self, ind1, ind2):
        """Calculate if ind1 is dominated by ind2 according to all objectives in objective_dict.

        If ind2 is better or equal to ind1 in all objectives, and strictly better than ind1 in at least one objective.

        """
        wins = []  # 1 dominates 2
        wins += [ind1.fitness > ind2.fitness]
        wins += [ind1.age < ind2.age]
        return not np.any(wins)

    def sort_by_objectives(self):
        """Sorts the population multiple times by each objective, from least important to most important."""
        self.sort(key="age", reverse=False)
        self.sort(key="fitness", reverse=True)

        self.sort(key="pareto_level", reverse=False)  # min

    def update_ages(self):
        """Increment the age of each individual."""
        for ind in self:
            ind.age += 1

    def sort(self, key, reverse=False):
        """Sort individuals by their attributes.

        Parameters
        ----------
        key : str
            An individual-level attribute.

        reverse : bool
            True sorts from largest to smallest (useful for maximizing an objective).
            False sorts from smallest to largest (useful for minimizing an objective).

        """
        return self.individuals.sort(reverse=reverse, key=operator.attrgetter(key))

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        return iter(self.individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        try:
            return n in self.individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        return len(self.individuals)

    def __getitem__(self, n):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.individuals[n]

    def pop(self, index=None):
        """Remove and return item at index (default last)."""
        return self.individuals.pop(index)

    def append(self, individuals):
        """Append a list of new individuals to the end of the population.

        Parameters
        ----------
        individuals : list of/or INDIVIDUAL
            A list of individuals to append or a single INDIVIDUAL to append

        """
        if type(individuals) == list:
            for n in range(len(individuals)):
                if type(individuals[n]) != INDIVIDUAL:
                    raise TypeError("Non-INDIVIDUAL added to the population")
            self.individuals += individuals
        elif type(individuals) == INDIVIDUAL:
            self.individuals += [individuals]
        else:
            raise TypeError("Non-INDIVIDUAL added to the population")




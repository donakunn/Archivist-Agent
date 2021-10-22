# probStochSim.py - Probabilistic inference using stochastic simulation
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
import matplotlib.pyplot as plt
from Localization.probGraphicalModels import InferenceMethod


def sample_one(dist):
    """returns the index of a single sample from normalized distribution dist."""
    rand = random.random() * sum(dist.values())
    cum = 0  # cumulative weights
    for v in dist:
        cum += dist[v]
        if cum > rand:
            return v


def sample_multiple(dist, num_samples):
    """returns a list of num_samples values selected using distribution dist.
    dist is a {value:weight} dictionary that does not need to be normalized
    """
    total = sum(dist.values())
    rands = sorted(random.random() * total for _ in range(num_samples))
    result = []
    dist_items = list(dist.items())
    cum = dist_items[0][1]  # cumulative sum
    index = 0
    for r in rands:
        while r > cum:
            index += 1
            cum += dist_items[index][1]
        result.append(dist_items[index][0])
    return result


def test_sampling(dist, num_samples):
    """Given a distribution, dist, draw num_samples samples
    and return the resulting counts
    """
    result = {v: 0 for v in dist}
    for v in sample_multiple(dist, num_samples):
        result[v] += 1
    return result


class SamplingInferenceMethod(InferenceMethod):
    """The abstract class of sampling-based belief network inference methods"""

    def __init__(self, gm=None):
        InferenceMethod.__init__(self, gm)

    def query(self, qvar, obs={}, number_samples=1000, sample_order=None):
        raise NotImplementedError("SamplingInferenceMethod query")  # abstract


class RejectionSampling(SamplingInferenceMethod):
    """The class that queries Graphical Models using Rejection Sampling.

    gm is a belief network to query
    """
    method_name = "rejection sampling"

    def __init__(self, gm=None):
        SamplingInferenceMethod.__init__(self, gm)

    def query(self, qvar, obs={}, number_samples=1000, sample_order=None):
        """computes P(qvar|obs) where
        qvar is a variable.
        obs is a {variable:value} dictionary.
        sample_order is a list of variables where the parents
          come before the variable.
        """
        if sample_order is None:
            sample_order = self.gm.topological_sort()
        self.display(2, *sample_order, sep="\t")
        counts = {val: 0 for val in qvar.domain}
        for i in range(number_samples):
            rejected = False
            sample = {}
            for nvar in sample_order:
                fac = self.gm.var2cpt[nvar]  # factor with nvar as child
                val = sample_one({v: fac.get_value(sample | {nvar: v}) for v in nvar.domain})
                self.display(2, val, end="\t")
                if nvar in obs and obs[nvar] != val:
                    rejected = True
                    self.display(2, "Rejected")
                    break
                sample[nvar] = val
            if not rejected:
                counts[sample[qvar]] += 1
                self.display(2, "Accepted")
        tot = sum(counts.values())
        # As well as the distribution we also include raw counts
        return {c: v / tot if tot > 0 else 1 / len(qvar.domain) for (c, v) in counts.items()} | {"raw_counts": counts}


class LikelihoodWeighting(SamplingInferenceMethod):
    """The class that queries Graphical Models using Likelihood weighting.

    gm is a belief network to query
    """
    method_name = "likelihood weighting"

    def __init__(self, gm=None):
        SamplingInferenceMethod.__init__(self, gm)

    def query(self, qvar, obs={}, number_samples=1000, sample_order=None):
        """computes P(qvar|obs) where
        qvar is a variable.
        obs is a {variable:value} dictionary.
        sample_order is a list of factors where factors defining the parents
          come before the factors for the child.
        """
        if sample_order is None:
            sample_order = self.gm.topological_sort()
        self.display(2, *[v for v in sample_order
                          if v not in obs], sep="\t")
        counts = {val: 0 for val in qvar.domain}
        for i in range(number_samples):
            sample = {}
            weight = 1.0
            for nvar in sample_order:
                fac = self.gm.var2cpt[nvar]
                if nvar in obs:
                    sample[nvar] = obs[nvar]
                    weight *= fac.get_value(sample)
                else:
                    val = sample_one({v: fac.get_value(sample | {nvar: v}) for v in nvar.domain})
                    self.display(2, val, end="\t")
                    sample[nvar] = val
            counts[sample[qvar]] += weight
            self.display(2, weight)
        tot = sum(counts.values())
        # as well as the distribution we also include the raw counts
        return {c: v / tot for (c, v) in counts.items()} | {"raw_counts": counts}


class ParticleFiltering(SamplingInferenceMethod):
    """The class that queries Graphical Models using Particle Filtering.

    gm is a belief network to query
    """
    method_name = "particle filtering"

    def __init__(self, gm=None):
        SamplingInferenceMethod.__init__(self, gm)

    def query(self, qvar, obs={}, number_samples=1000, sample_order=None):
        """computes P(qvar|obs) where
        qvar is a variable.
        obs is a {variable:value} dictionary.
        sample_order is a list of factors where factors defining the parents
          come before the factors for the child.
        """
        if sample_order is None:
            sample_order = self.gm.topological_sort()
        self.display(2, *[v for v in sample_order
                          if v not in obs], sep="\t")
        particles = [{} for i in range(number_samples)]
        for nvar in sample_order:
            fac = self.gm.var2cpt[nvar]
            if nvar in obs:
                weights = [fac.get_value(part | {nvar: obs[nvar]}) for part in particles]
                particles = [p | {nvar: obs[nvar]} for p in resample(particles, weights, number_samples)]
            else:
                for part in particles:
                    part[nvar] = sample_one({v: fac.get_value(part | {nvar: v}) for v in nvar.domain})
                self.display(2, part[nvar], end="\t")
        counts = {val: 0 for val in qvar.domain}
        for part in particles:
            counts[part[qvar]] += 1
        tot = sum(counts.values())
        # as well as the distribution we also include the raw counts
        return {c: v / tot for (c, v) in counts.items()} | {"raw_counts": counts}


def resample(particles, weights, num_samples):
    """returns num_samples copies of particles resampled according to weights.
    particles is a list of particles
    weights is a list of positive numbers, of same length as particles
    num_samples is n integer
    """
    total = sum(weights)
    rands = sorted(random.random() * total for i in range(num_samples))
    result = []
    cum = weights[0]  # cumulative sum
    index = 0
    for r in rands:
        while r > cum:
            index += 1
            cum += weights[index]
        result.append(particles[index])
    return result


class GibbsSampling(SamplingInferenceMethod):
    """The class that queries Graphical Models using Gibbs Sampling.

    bn is a graphical model (e.g., a belief network) to query
    """
    method_name = "Gibbs sampling"

    def __init__(self, gm=None):
        SamplingInferenceMethod.__init__(self, gm)
        self.gm = gm

    def query(self, qvar, obs={}, number_samples=1000, burn_in=100, sample_order=None):
        """computes P(qvar|obs) where
        qvar is a variable.
        obs is a {variable:value} dictionary.
        sample_order is a list of non-observed variables in order, or
        if sample_order None, the variables are shuffled at each iteration.
        """
        counts = {val: 0 for val in qvar.domain}
        if sample_order is not None:
            variables = sample_order
        else:
            variables = [v for v in self.gm.variables if v not in obs]
        var_to_factors = {v: set() for v in self.gm.variables}
        for fac in self.gm.factors:
            for var in fac.variables:
                var_to_factors[var].add(fac)
        sample = {var: random.choice(var.domain) for var in variables}
        self.display(2, "Sample:", sample)
        sample.update(obs)
        for i in range(burn_in + number_samples):
            if sample_order is None:
                random.shuffle(variables)
            for var in variables:
                # get unnormalized probability distribution of var given its neighbours
                vardist = {val: 1 for val in var.domain}
                for val in var.domain:
                    sample[var] = val
                    for fac in var_to_factors[var]:  # Markov blanket
                        vardist[val] *= fac.get_value(sample)
                sample[var] = sample_one(vardist)
            if i >= burn_in:
                counts[sample[qvar]] += 1
        tot = sum(counts.values())
        # as well as the computed distribution, we also include raw counts
        return {c: v / tot for (c, v) in counts.items()} | {"raw_counts": counts}


def plot_stats(method, qvar, qval, obs, number_runs=1000, **queryargs):
    """Plots a cumulative distribution of the prediction of the model.
    method is a InferenceMethod (that implements appropriate query(.))
    plots P(qvar=qval | obs)
    qvar is the query variable, qval is corresponding value
    obs is the {variable:value} dictionary representing the observations
    number_iterations is the number of runs that are plotted
    **queryargs is the arguments to query (often number_samples for sampling methods)
    """
    plt.ion()
    plt.xlabel("value")
    plt.ylabel("Cumulative Number")
    method.max_display_level, prev_mdl = 0, method.max_display_level  # no display
    answers = [method.query(qvar, obs, **queryargs)
               for i in range(number_runs)]
    values = [ans[qval] for ans in answers]
    label = f"{method.method_name} P({qvar}={qval}|{','.join(f'{var}={val}' for (var, val) in obs.items())})"
    values.sort()
    plt.plot(values, range(number_runs), label=label)
    plt.legend()  # loc="upper left")
    plt.draw()
    method.max_display_level = prev_mdl  # restore display level


def plot_mult(methods, example, qvar, qval, obs, number_samples=1000, number_runs=1000):
    for method in methods:
        solver = method(example)
        if isinstance(method, SamplingInferenceMethod):
            plot_stats(solver, qvar, qval, obs, number_samples, number_runs)
        else:
            plot_stats(solver, qvar, qval, obs, number_runs)

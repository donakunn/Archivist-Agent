# probHMM.py - Hidden Markov Model
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
from Localization.probStochSim import sample_one
from Localization.probStochSim import resample


class HMM(object):
    def __init__(self, states, obsvars, pobs, trans, indist):
        """A hidden Markov model.
        states - set of states
        obsvars - set of observation variables
        pobs - probability of observations, pobs[i][s] is P(Obs_i=True | State=s)
        trans - transition probability - trans[i][j] gives P(State=j | State=i)
        indist - initial distribution - indist[s] is P(State_0 = s)
        """
        self.states = states
        self.obsvars = obsvars
        self.pobs = pobs
        self.trans = trans
        self.indist = indist


class HMMVEfilter(Displayable):
    def __init__(self, hmm):
        self.hmm = hmm
        self.state_dist = hmm.indist

    def filter(self, obsseq):
        """updates and returns the state distribution following the sequence of
        observations in obsseq using variable elimination.

        Note that it first advances time.
        This is what is required if it is called sequentially.
        If that is not what is wanted initially, do an observe first.
        """
        for obs in obsseq:
            self.advance()  # advance time
            self.observe(obs)  # observe
        return self.state_dist

    def observe(self, obs):
        """updates state conditioned on observations.
        obs is a list of values for each observation variable"""
        for i in self.hmm.obsvars:
            self.state_dist = {st: self.state_dist[st] * (self.hmm.pobs[i][st]
                                                          if obs[i] else (1 - self.hmm.pobs[i][st]))
                               for st in self.hmm.states}
        norm = sum(self.state_dist.values())  # normalizing constant
        self.state_dist = {st: self.state_dist[st] / norm for st in self.hmm.states}
        self.display(2, "After observing", obs, "state distribution:", self.state_dist)

    def advance(self):
        """advance to the next time"""
        nextstate = {st: 0.0 for st in self.hmm.states}  # distribution over next states
        for j in self.hmm.states:  # j ranges over next states
            for i in self.hmm.states:  # i ranges over previous states
                nextstate[j] += self.hmm.trans[i][j] * self.state_dist[i]
        self.state_dist = nextstate
        self.display(2, "After advancing state distribution:", self.state_dist)


class HMMparticleFilter(Displayable):
    def __init__(self, hmm, number_particles=1000):
        self.hmm = hmm
        self.particles = [sample_one(hmm.indist)
                          for _ in range(number_particles)]
        self.weights = [1 for _ in range(number_particles)]
        self.first_time = True

    def filter(self, obsseq):
        """returns the state distribution following the sequence of
        observations in obsseq using particle filtering. 

        Note that it first advances time.
        This is what is required if it is called after previous filtering.
        If that is not what is wanted initially, do an observe first.
        """
        for obs in obsseq:
            if not self.first_time:
                self.advance()  # advance time
            self.observe(obs)  # observe
            self.resample_particles()
        self.first_time = False
        return self.histogram()

    def advance(self):
        """advance to the next time.
        This assumes that all of the weights are 1."""
        self.particles = [sample_one(self.hmm.trans[st])
                          for st in self.particles]

    def observe(self, obs):
        """reweighs the particles to incorporate observations obs"""
        for i in range(len(self.particles)):
            for obv in obs:
                if obs[obv]:
                    if self.particles[i] in self.hmm.pobs[obv]:
                        self.weights[i] *= self.hmm.pobs[obv][self.particles[i]]
                elif self.particles[i] in self.hmm.pobs[obv]:
                    self.weights[i] *= 1 - self.hmm.pobs[obv][self.particles[i]]

    def histogram(self):
        """returns list of the probability of each state as represented by
        the particles"""
        tot = 0
        hist = {st: 0.0 for st in self.hmm.states}
        for (st, wt) in zip(self.particles, self.weights):
            hist[st] += wt
            tot += wt
        return {st: hist[st] / tot for st in hist}

    def resample_particles(self):
        """resamples to give a new set of particles."""
        self.particles = resample(self.particles, self.weights, len(self.particles))
        self.weights = [1] * len(self.particles)


def simulate(hmm, horizon):
    """returns a pair of (state sequence, observation sequence) of length horizon.
    for each time t, the agent is in state_sequence[t] and
    observes observation_sequence[t]
    """
    state = sample_one(hmm.indist)
    obsseq = []
    stateseq = []
    for time in range(horizon):
        stateseq.append(state)
        newobs = {obs: sample_one({0: 1 - hmm.pobs[obs][state], 1: hmm.pobs[obs][state]})
                  for obs in hmm.obsvars}
        obsseq.append(newobs)
        state = sample_one(hmm.trans[state])
    return stateseq, obsseq


def simobs(hmm, stateseq):
    """returns observation sequence for the state sequence"""
    obsseq = []
    for state in stateseq:
        newobs = {obs: sample_one({0: 1 - hmm.pobs[obs][state], 1: hmm.pobs[obs][state]})
                  for obs in hmm.obsvars}
        obsseq.append(newobs)
    return obsseq


def create_eg(hmm, n):
    """Create an annotated example for horizon n"""
    seq, obs = simulate(hmm, n)
    print("True state sequence:", seq)
    print("Sequence of observations:\n", obs)
    hmmfilter = HMMVEfilter(hmm)
    dist = hmmfilter.filter(obs)
    print("Resulting distribution over states:\n", dist)

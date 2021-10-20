from Localization.probHMM import HMM, HMMparticleFilter
class Localizator:

    def __init__(self):
        self.HMM_with_sampling = None

    def build_archive_HMM(self):
        obs = {'m1', 'm2', 'm3'}
        states = {'middle', 'c1', 'c2', 'c3'}
        # pobs gives the observation model:
        # pobs[mi][state] is P(mi=on | state)
        closeMic = 0.6
        farMic = 0.1
        midMic = 0.4
        pobs = {'m1': {'middle': midMic, 'c1': closeMic, 'c2': farMic, 'c3': farMic},  # mic 1
                'm2': {'middle': midMic, 'c1': farMic, 'c2': closeMic, 'c3': farMic},  # mic 2
                'm3': {'middle': midMic, 'c1': farMic, 'c2': farMic, 'c3': closeMic}}  # mic 3
        # trans specifies the dynamics
        # trans[i] is the distribution over states resulting from state i
        # trans[i][j] gives P(S=j | S=i)
        sm = 0.7
        mmc = 0.1  # transition probabilities when in middle
        sc = 0.8
        mcm = 0.1
        mcc = 0.05  # transition probabilities when in a corner
        trans = {'middle': {'middle': sm, 'c1': mmc, 'c2': mmc, 'c3': mmc},  # was in middle
                 'c1': {'middle': mcm, 'c1': sc, 'c2': mcc, 'c3': mcc},  # was in corner 1
                 'c2': {'middle': mcm, 'c1': mcc, 'c2': sc, 'c3': mcc},  # was in corner 2
                 'c3': {'middle': mcm, 'c1': mcc, 'c2': mcc, 'c3': sc}}  # was in corner 3
        # initially we have a uniform distribution over the animal's state
        indist = {st: 1.0 / len(states) for st in states}
        self.HMM_with_sampling = HMMparticleFilter(HMM(states, obs, pobs, trans, indist))

    def test(self):
        HMMparticleFilter.max_display_level = 2  # show each step
        self.HMM_with_sampling.filter([{'m1': 0, 'm2': 1, 'm3': 1}, {'m1': 1, 'm2': 0, 'm3': 1}])
        hmm1pf2 = HMMparticleFilter(self.HMM_with_sampling)
        hmm1pf2.filter([{'m1': 1, 'm2': 0}, {'m1': 0, 'm2': 1, 'm3': 0}, {'m1': 1, 'm2': 0, 'm3': 0},
                        {'m1': 0, 'm2': 0, 'm3': 0}, {'m1': 0, 'm2': 0, 'm3': 0}, {'m1': 0, 'm2': 0, 'm3': 0},
                        {'m1': 0, 'm2': 0, 'm3': 0}, {'m1': 0, 'm2': 0, 'm3': 1}, {'m1': 0, 'm2': 0, 'm3': 1},
                        {'m1': 0, 'm2': 0, 'm3': 1}])
        hmm1pf3 = HMMparticleFilter(self.HMM_with_sampling)
        hmm1pf3.filter([{'m1': 1, 'm2': 0, 'm3': 0}, {'m1': 0, 'm2': 0, 'm3': 0}, {'m1': 1, 'm2': 0, 'm3': 0},
                        {'m1': 1, 'm2': 0, 'm3': 1}])

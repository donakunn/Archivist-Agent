from Localization.probHMM import HMM, HMMparticleFilter


class Localizator:

    def __init__(self):
        self.HMM_with_sampling = None

    def build_archive_HMM(self):
        obs = {'s_PR1', 's_PR2', 's_PR3', 's_PR4', 's_C1', 's_C2', 's_C3', 's_C4', 's_C5', 's_C6', 's_C7', 's_C8',
               's_C9', 's_C10', 's_C11', 's_alt.atheism', 's_comp.graphics', 's_comp.os.ms-windows.misc',
               's_comp.sys.ibm.pc.hardware', 's_comp.sys.mac.hardware', 's_comp.windows.x', 'misc.forsale',
               's_rec.autos', 's_rec.motorcycles', 's_rec.sport.baseball', 's_rec.sport.hockey', 'sci.crypt',
               's_sci.electronics', 's_sci.med', 's_sci.space', 's_soc.religion.christian',
               's_talk.politics.guns', 's_talk.politics.mideast', 's_talk.politics.misc',
               's_talk.religion.misc'}
        states = {'PR1', 'PR2', 'PR3', 'PR4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                  'C9', 'C10', 'C11', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                  'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
                  'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'}
        # pobs gives the observation model:
        # pobs[mi][state] is P(mi=on | state)
        actual_state_sensor = 0.7
        close_state_sensor = 0.1

        pobs = {'s_PR1': {'PR1': actual_state_sensor, 'C1': close_state_sensor},  # mic 1
                's_PR2': {'PR2': actual_state_sensor, 'C4': close_state_sensor},  # mic 2
                's_PR3': {'PR3': actual_state_sensor, 'C11': close_state_sensor},
                's_PR4': {'PR4': actual_state_sensor, 'C8': close_state_sensor},
                's_C1': {'C1': actual_state_sensor, 'alt.atheism': close_state_sensor, 'C2': close_state_sensor,
                         'C5': close_state_sensor, 'PR1': close_state_sensor, 'talk.religion.misc': close_state_sensor},
                's_C2': {'C2': actual_state_sensor, 'comp.graphics': close_state_sensor, 'C3': close_state_sensor,
                         'C6': close_state_sensor, 'C5': close_state_sensor,
                         'comp.os.ms-windows.misc': close_state_sensor, 'C1': close_state_sensor},
                's_C3': {'C3': actual_state_sensor, 'comp.os.ms-windows.misc': close_state_sensor,
                         'C2': close_state_sensor,
                         'comp.sys.ibm.pc.hardware': close_state_sensor, 'C4': close_state_sensor,
                         'C7': close_state_sensor, 'C6': close_state_sensor},
                's_C4': {'C4': actual_state_sensor, 'comp.sys.mac.hardware': close_state_sensor,
                         'comp.sys.ibm.pc.hardware': close_state_sensor, 'PR2': close_state_sensor,
                         'comp.windows.x': close_state_sensor, 'misc.forsale': close_state_sensor,
                         'C7': close_state_sensor, 'C3': close_state_sensor},
                's_C7': {'C7': actual_state_sensor, 'rec.autos': close_state_sensor, 'C11': close_state_sensor,
                         'C10': close_state_sensor, 'C6': close_state_sensor, 'C3': close_state_sensor,
                         'C4': close_state_sensor},
                's_C6': {'C6': actual_state_sensor, 'C2': close_state_sensor, 'C3': close_state_sensor,
                         'C7': close_state_sensor, 'C10': close_state_sensor, 'C9': close_state_sensor,
                         'C5': close_state_sensor},
                's_C5': {'C5': actual_state_sensor, 'talk.politics.misc': close_state_sensor, 'C1': close_state_sensor,
                         'C2': close_state_sensor, 'C6': close_state_sensor, 'C9': close_state_sensor,
                         'C8': close_state_sensor, 'talk.politics.guns': close_state_sensor,
                         'talk.politics.mideast': close_state_sensor},
                's_C8': {'C8': actual_state_sensor, 'C5': close_state_sensor, 'C9': close_state_sensor,
                         'sci.med': close_state_sensor, 'sci.space': close_state_sensor, 'PR4': close_state_sensor,
                         'soc.religion.christian': close_state_sensor},
                's_C9': {'C9': actual_state_sensor, 'C8': close_state_sensor, 'C5': close_state_sensor,
                         'C6': close_state_sensor, 'C10': close_state_sensor, 'sci.electronics': close_state_sensor,
                         'sci.med': close_state_sensor},
                's_C10': {'C10': actual_state_sensor, 'C6': close_state_sensor, 'C7': close_state_sensor,
                          'C11': close_state_sensor, 'sci.crypt': close_state_sensor,
                          'sci.electronics': close_state_sensor, 'C9': close_state_sensor},
                's_C11': {'C11': actual_state_sensor, 'rec.motorcycles': close_state_sensor,
                          'rec.sport.baseball': close_state_sensor, 'PR3': close_state_sensor,
                          'rec.sport.hockey': close_state_sensor, 'sci.crypt': close_state_sensor,
                          'C10': close_state_sensor, 'C7': close_state_sensor},
                's_alt.atheism': {'alt.atheism': actual_state_sensor, 'C1': close_state_sensor},
                's_comp.graphics': {'comp.graphics': actual_state_sensor, 'C2': close_state_sensor},
                's_comp.os.ms-windows.misc': {'comp.os.ms-windows.misc': actual_state_sensor, 'C2': close_state_sensor,
                                              'C3': close_state_sensor},
                's_comp.sys.ibm.pc.hardware': {'comp.sys.ibm.pc.hardware': actual_state_sensor, 'C3': close_state_sensor
                                               , 'C4': close_state_sensor},
                's_comp.sys.mac.hardware': {'comp.sys.mac.hardware': actual_state_sensor, 'C4': close_state_sensor},
                's_comp.windows.x': {'comp.windows.x': actual_state_sensor, 'C4': close_state_sensor},
                's_misc.forsale': {'misc.forsale': actual_state_sensor, 'C4': close_state_sensor},
                's_rec.autos': {'rec.autos': actual_state_sensor, 'C7': close_state_sensor},
                's_rec.motorcycles': {'rec.motorcycles': actual_state_sensor, 'C11': close_state_sensor},
                's_rec.sport.baseball': {'rec.sport.baseball': actual_state_sensor, 'C11': close_state_sensor},
                's_rec.sport.hockey': {'rec.sport.hockey': actual_state_sensor, 'C11': close_state_sensor},
                's_sci.crypt': {'sci.crypt': actual_state_sensor, 'C11': close_state_sensor,
                                'C10': close_state_sensor},
                's_sci.electronics': {'sci.electronics': actual_state_sensor, 'C10': close_state_sensor,
                                      'C9': close_state_sensor},
                's_sci.med': {'sci.med': actual_state_sensor, 'C9': close_state_sensor, 'C8': close_state_sensor},
                's_sci.space': {'sci.space': actual_state_sensor, 'C8': close_state_sensor},
                's_soc.religion.christian': {'soc.religion.christian': actual_state_sensor, 'C8': close_state_sensor},
                's_talk.politics.guns': {'talk.politics.guns': actual_state_sensor, 'C5': close_state_sensor},
                's_talk.politics.mideast': {'talk.politics.mideast': actual_state_sensor, 'C5': close_state_sensor},
                's_talk.politics.misc': {'talk.politics.misc': actual_state_sensor, 'C5': close_state_sensor},
                's_talk.religion.misc': {'talk.religion.misc': actual_state_sensor, 'C1': close_state_sensor}
                }  # mic 3
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

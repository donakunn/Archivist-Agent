import random

from Localization.probHMM import HMM, HMMparticleFilter


class Localizator:

    def __init__(self):
        self.HMM_with_sampling = None
        self.obs = {}
        self.states = {}

    def build_archive_HMM(self):
        self.obs = {'s_PR1', 's_PR2', 's_PR3', 's_PR4', 's_C1', 's_C2', 's_C3', 's_C4', 's_C5', 's_C6', 's_C7', 's_C8',
                    's_C9', 's_C10', 's_C11', 's_alt.atheism', 's_comp.graphics', 's_comp.os.ms-windows.misc',
                    's_comp.sys.ibm.pc.hardware', 's_comp.sys.mac.hardware', 's_comp.windows.x', 's_misc.forsale',
                    's_rec.autos', 's_rec.motorcycles', 's_rec.sport.baseball', 's_rec.sport.hockey', 's_sci.crypt',
                    's_sci.electronics', 's_sci.med', 's_sci.space', 's_soc.religion.christian',
                    's_talk.politics.guns', 's_talk.politics.mideast', 's_talk.politics.misc',
                    's_talk.religion.misc'}
        self.states = {'PR1', 'PR2', 'PR3', 'PR4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                       'C9', 'C10', 'C11', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                       'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
                       'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                       'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
                       'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'}
        actual_state_sensor = 0.95
        close_state_sensor = 0.05

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
                's_comp.sys.ibm.pc.hardware': {'comp.sys.ibm.pc.hardware': actual_state_sensor,
                                               'C3': close_state_sensor,
                                               'C4': close_state_sensor},
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
                }
        s_pr = 0.1
        s_c = 0.05
        trans = {'PR1': {'PR1': s_pr, 'C1': 1 - s_pr},
                 'PR2': {'PR2': s_pr, 'C4': 1 - s_pr},
                 'PR3': {'PR3': s_pr, 'C11': 1 - s_pr},
                 'PR4': {'PR4': s_pr, 'C8': 1 - s_pr},
                 'C1': {'C1': s_c, 'alt.atheism': (1 - s_c) / 5, 'C2': (1 - s_c) / 5,
                        'C5': (1 - s_c) / 5, 'PR1': (1 - s_c) / 5,
                        'talk.religion.misc': (1 - s_c) / 5},
                 'C2': {'C2': s_c, 'comp.graphics': (1 - s_c) / 6, 'C3': (1 - s_c) / 6,
                        'C6': (1 - s_c) / 6, 'C5': (1 - s_c) / 6,
                        'comp.os.ms-windows.misc': (1 - s_c) / 6, 'C1': (1 - s_c) / 6},
                 'C3': {'C3': s_c, 'comp.os.ms-windows.misc': (1 - s_c) / 6,
                        'C2': (1 - s_c) / 6,
                        'comp.sys.ibm.pc.hardware': (1 - s_c) / 6, 'C4': (1 - s_c) / 6,
                        'C7': (1 - s_c) / 6, 'C6': (1 - s_c) / 6},
                 'C4': {'C4': s_c, 'comp.sys.mac.hardware': (1 - s_c) / 7,
                        'comp.sys.ibm.pc.hardware': (1 - s_c) / 7, 'PR2': (1 - s_c) / 7,
                        'comp.windows.x': (1 - s_c) / 7, 'misc.forsale': (1 - s_c) / 7,
                        'C7': (1 - s_c) / 7, 'C3': (1 - s_c) / 7},
                 'C7': {'C7': s_c, 'rec.autos': (1 - s_c) / 6, 'C11': (1 - s_c) / 6,
                        'C10': (1 - s_c) / 6, 'C6': (1 - s_c) / 6, 'C3': (1 - s_c) / 6,
                        'C4': (1 - s_c) / 6},
                 'C6': {'C6': s_c, 'C2': (1 - s_c) / 6, 'C3': (1 - s_c) / 6,
                        'C7': (1 - s_c) / 6, 'C10': (1 - s_c) / 6, 'C9': (1 - s_c) / 6,
                        'C5': (1 - s_c) / 6},
                 'C5': {'C5': s_c, 'talk.politics.misc': (1 - s_c) / 8, 'C1': (1 - s_c) / 8,
                        'C2': (1 - s_c) / 8, 'C6': (1 - s_c) / 8, 'C9': (1 - s_c) / 8,
                        'C8': (1 - s_c) / 8, 'talk.politics.guns': (1 - s_c) / 8,
                        'talk.politics.mideast': (1 - s_c) / 8},
                 'C8': {'C8': s_c, 'C5': (1 - s_c) / 6, 'C9': (1 - s_c) / 6,
                        'sci.med': (1 - s_c) / 6, 'sci.space': (1 - s_c) / 6, 'PR4': (1 - s_c) / 6,
                        'soc.religion.christian': (1 - s_c) / 6},
                 'C9': {'C9': s_c, 'C8': (1 - s_c) / 6, 'C5': (1 - s_c) / 6,
                        'C6': (1 - s_c) / 6, 'C10': (1 - s_c) / 6, 'sci.electronics': (1 - s_c) / 6,
                        'sci.med': (1 - s_c) / 6},
                 'C10': {'C10': s_c, 'C6': (1 - s_c) / 6, 'C7': (1 - s_c) / 6,
                         'C11': (1 - s_c) / 6, 'sci.crypt': (1 - s_c) / 6,
                         'sci.electronics': (1 - s_c) / 6, 'C9': (1 - s_c) / 6},
                 'C11': {'C11': s_c, 'rec.motorcycles': (1 - s_c) / 7,
                         'rec.sport.baseball': (1 - s_c) / 7, 'PR3': (1 - s_c) / 7,
                         'rec.sport.hockey': (1 - s_c) / 7, 'sci.crypt': (1 - s_c) / 7,
                         'C10': (1 - s_c) / 7, 'C7': (1 - s_c) / 7},
                 'alt.atheism': {'alt.atheism': s_c, 'C1': 1 - s_c},
                 'comp.graphics': {'comp.graphics': s_c, 'C2': 1 - s_c},
                 'comp.os.ms-windows.misc': {'comp.os.ms-windows.misc': s_c, 'C2': (1 - s_c) / 2,
                                             'C3': (1 - s_c) / 2},
                 'comp.sys.ibm.pc.hardware': {'comp.sys.ibm.pc.hardware': s_c,
                                              'C3': (1 - s_c) / 2, 'C4': (1 - s_c) / 2},
                 'comp.sys.mac.hardware': {'comp.sys.mac.hardware': s_c, 'C4': 1 - s_c},
                 'comp.windows.x': {'comp.windows.x': s_c, 'C4': 1 - s_c},
                 'misc.forsale': {'misc.forsale': s_c, 'C4': 1 - s_c},
                 'rec.autos': {'rec.autos': s_c, 'C7': 1 - s_c},
                 'rec.motorcycles': {'rec.motorcycles': s_c, 'C11': 1 - s_c},
                 'rec.sport.baseball': {'rec.sport.baseball': s_c, 'C11': 1 - s_c},
                 'rec.sport.hockey': {'rec.sport.hockey': s_c, 'C11': 1 - s_c},
                 'sci.crypt': {'sci.crypt': s_c, 'C11': (1 - s_c) / 2,
                               'C10': (1 - s_c) / 2},
                 'sci.electronics': {'sci.electronics': s_c, 'C10': (1 - s_c) / 2,
                                     'C9': (1 - s_c) / 2},
                 'sci.med': {'sci.med': s_c, 'C9': (1 - s_c) / 2, 'C8': (1 - s_c) / 2},
                 'sci.space': {'sci.space': s_c, 'C8': 1 - s_c},
                 'soc.religion.christian': {'soc.religion.christian': s_c, 'C8': 1 - s_c},
                 'talk.politics.guns': {'talk.politics.guns': s_c, 'C5': 1 - s_c},
                 'talk.politics.mideast': {'talk.politics.mideast': s_c, 'C5': 1 - s_c},
                 'talk.politics.misc': {'talk.politics.misc': s_c, 'C5': 1 - s_c},
                 'talk.religion.misc': {'talk.religion.misc': s_c, 'C1': 1 - s_c}
                 }
        indist = {st: 1.0 / len(self.states) for st in self.states}
        self.HMM_with_sampling = HMMparticleFilter(HMM(self.states, self.obs, pobs, trans, indist))

    def build_observation(self, state):
        state = 's_' + state
        observation = {}
        for sensor in self.obs:
            if state == sensor:
                observation[sensor] = 1
            elif random.uniform(0, 1) > 0.05:
                observation[sensor] = 0
        return observation

    def observe(self, state):
        return self.HMM_with_sampling.filter([self.build_observation(state)])

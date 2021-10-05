from Search.searchBranchAndBound import DF_branch_and_bound
from Search.searchProblem import Search_problem_from_explicit_graph, Arc


class ArchivePathSearcher:
    def __init__(self):
        self.cyclic_delivery_problem = Search_problem_from_explicit_graph(
            {'mail', 'ts', 'o103', 'o109', 'o111', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2', 'c3',
             'o125', 'o123', 'o119', 'r123', 'storage'},
            [Arc('ts', 'mail', 6), Arc('mail', 'ts', 6),
             Arc('o103', 'ts', 8), Arc('ts', 'o103', 8),
             Arc('o103', 'b3', 4),
             Arc('o103', 'o109', 12), Arc('o109', 'o103', 12),
             Arc('o109', 'o119', 16), Arc('o119', 'o109', 16),
             Arc('o109', 'o111', 4), Arc('o111', 'o109', 4),
             Arc('b1', 'c2', 3),
             Arc('b1', 'b2', 6), Arc('b2', 'b1', 6),
             Arc('b2', 'b4', 3), Arc('b4', 'b2', 3),
             Arc('b3', 'b1', 4), Arc('b1', 'b3', 4),
             Arc('b3', 'b4', 7), Arc('b4', 'b3', 7),
             Arc('b4', 'o109', 7),
             Arc('c1', 'c3', 8), Arc('c3', 'c1', 8),
             Arc('c2', 'c3', 6), Arc('c3', 'c2', 6),
             Arc('c2', 'c1', 4), Arc('c1', 'c2', 4),
             Arc('o123', 'o125', 4), Arc('o125', 'o123', 4),
             Arc('o123', 'r123', 4), Arc('r123', 'o123', 4),
             Arc('o119', 'o123', 9), Arc('o123', 'o119', 9),
             Arc('o119', 'storage', 7), Arc('storage', 'o119', 7)],
            start='o103',
            goals={'b4'},
            hmap={
                'mail': 26,
                'ts': 23,
                'o103': 21,
                'o109': 24,
                'o111': 27,
                'o119': 11,
                'o123': 4,
                'o125': 6,
                'r123': 0,
                'b1': 13,
                'b2': 15,
                'b3': 17,
                'b4': 18,
                'c1': 6,
                'c2': 10,
                'c3': 12,
                'storage': 12
            }
        )

    def goal_searcher_with_branch_and_bound(self):
        print(DF_branch_and_bound(self.cyclic_delivery_problem).search())


import random
from Search.searchBranchAndBound import DF_branch_and_bound
from Search.searchProblem import Search_problem_from_explicit_graph, Arc


class ArchivePathSearcher:
    def __init__(self):
        self.current_position = random.choice(['PR1', 'PR2', 'PR3', 'PR4'])
        self.cyclic_delivery_problem = Search_problem_from_explicit_graph(
            {'PR1', 'PR2', 'PR3', 'PR4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
             'C9', 'C10', 'C11', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
             'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
             'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
             'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
             'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'},
            [Arc('PR1', 'C1'), Arc('C1', 'PR1'),
             Arc('C1', 'alt.atheism'), Arc('alt.atheism', 'C1'),
             Arc('C1', 'C2'), Arc('C2', 'C1'),
             Arc('C1', 'C5'), Arc('C5', 'C1'),
             Arc('C1', 'talk.religion.misc'), Arc('talk.religion.misc', 'C1'),
             Arc('C2', 'comp.graphics'), Arc('comp.graphics', 'C2'),
             Arc('C2', 'comp.os.ms-windows.misc'), Arc('comp.os.ms-windows.misc', 'C2'),
             Arc('C2', 'C3'), Arc('C3', 'C2'),
             Arc('C2', 'C6'), Arc('C6', 'C2'),
             Arc('C2', 'C5'), Arc('C5', 'C2'),
             Arc('C3', 'comp.os.ms-windows.misc'), Arc('comp.os.ms-windows.misc', 'C3'),
             Arc('C3', 'comp.sys.ibm.pc.hardware'), Arc('comp.sys.ibm.pc.hardware', 'C3'),
             Arc('C3', 'C4'), Arc('C4', 'C3'),
             Arc('C3', 'C7'), Arc('C7', 'C3'),
             Arc('C3', 'C6'), Arc('C6', 'C3'),
             Arc('C4', 'comp.sys.ibm.pc.hardware'), Arc('comp.sys.ibm.pc.hardware', 'C4'),
             Arc('C4', 'comp.sys.mac.hardware'), Arc('comp.sys.mac.hardware', 'C4'),
             Arc('C4', 'PR2'), Arc('PR2', 'C4'),
             Arc('C4', 'comp.windows.x'), Arc('comp.windows.x', 'C4'),
             Arc('C4', 'misc.forsale'), Arc('misc.forsale', 'C4'),
             Arc('C4', 'C7'), Arc('C7', 'C4'),
             Arc('C7', 'rec.autos'), Arc('rec.autos', 'C7'),
             Arc('C7', 'C11'), Arc('C11', 'C7'),
             Arc('C7', 'C10'), Arc('C10', 'C7'),
             Arc('C7', 'C6'), Arc('C6', 'C7'),
             Arc('C6', 'C10'), Arc('C10', 'C6'),
             Arc('C6', 'C9'), Arc('C9', 'C6'),
             Arc('C6', 'C5'), Arc('C5', 'C6'),
             Arc('C5', 'C9'), Arc('C9', 'C5'),
             Arc('C5', 'C8'), Arc('C8', 'C5'),
             Arc('C5', 'talk.politics.guns'), Arc('talk.politics.guns', 'C5'),
             Arc('C5', 'talk.politics.mideast'), Arc('talk.politics.mideast', 'C5'),
             Arc('C5', 'talk.politics.misc'), Arc('talk.politics.misc', 'C5'),
             Arc('C8', 'soc.religion.christian'), Arc('soc.religion.christian', 'C8'),
             Arc('C8', 'PR4'), Arc('PR4', 'C8'),
             Arc('C8', 'sci.space'), Arc('sci.space', 'C8'),
             Arc('C8', 'sci.med'), Arc('sci.med', 'C8'),
             Arc('C8', 'C9'), Arc('C9', 'C8'),
             Arc('C9', 'sci.med'), Arc('sci.med', 'C9'),
             Arc('C9', 'sci.electronics'), Arc('sci.electronics', 'C9'),
             Arc('C9', 'C10'), Arc('C10', 'C9'),
             Arc('C10', 'sci.electronics'), Arc('sci.electronics', 'C10'),
             Arc('C10', 'sci.crypt'), Arc('sci.crypt', 'C10'),
             Arc('C10', 'C11'), Arc('C11', 'C10'),
             Arc('C11', 'sci.crypt'), Arc('sci.crypt', 'C11'),
             Arc('C11', 'rec.sport.hockey'), Arc('rec.sport.hockey', 'C11'),
             Arc('C11', 'PR3'), Arc('PR3', 'C11'),
             Arc('C11', 'rec.sport.baseball'), Arc('rec.sport.baseball', 'C11'),
             Arc('C11', 'rec.motorcycles'), Arc('C4', 'rec.motorcycles')],
            start=self.current_position
        )

    def heuristic_builder(self, node, target_position):
        already_discovered = [node]
        self.cyclic_delivery_problem.hmap.update({str(node): 0})  # la distanza fino a se stesso è 0
        while already_discovered:
            current_node = already_discovered.pop()
            for current_neighbor in self.cyclic_delivery_problem.neighbor_nodes(current_node):
                if current_neighbor not in self.cyclic_delivery_problem.hmap:
                    if (str(current_node)[0] == 'P' and str(current_neighbor)[0] == 'C') or (
                            str(current_node)[0] == 'C' and str(current_neighbor)[0] == 'P'):
                        self.cyclic_delivery_problem.hmap.update({str(current_neighbor): self.cyclic_delivery_problem
                                                                 .hmap.get(current_node) + 3})
                    elif str(current_node)[0] == 'C' and str(current_neighbor)[0] == 'C':
                        self.cyclic_delivery_problem.hmap.update({str(current_neighbor): self.cyclic_delivery_problem
                                                                 .hmap.get(current_node) + 8})
                    else:
                        self.cyclic_delivery_problem.hmap.update({str(current_neighbor): self.cyclic_delivery_problem
                                                                 .hmap.get(current_node) + 5})
                    already_discovered.append(current_neighbor)
        self.cyclic_delivery_problem.goals = {target_position}

    def goal_searcher_with_branch_and_bound(self, target_position):
        self.heuristic_builder(self.current_position, target_position)
        bound = 50
        path_searcher = DF_branch_and_bound(self.cyclic_delivery_problem, bound)   # 50 stima lunghezza massima percorso
        found_path = path_searcher.search()         ### bound viene settato come somma delle lunghezze degli archi
        best_path = None
        if found_path is not None:
            print(found_path)
            best_path = found_path
            bound = path_searcher.bound
            while found_path is not None:
                bound -= 1
                found_path = DF_branch_and_bound(self.cyclic_delivery_problem, bound).search()
                if found_path is not None:
                    print(found_path)
                    best_path = found_path
        return best_path

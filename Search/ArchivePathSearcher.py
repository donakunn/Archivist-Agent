import random
from Search.searchBranchAndBound import DF_branch_and_bound
from Search.searchProblem import Search_problem_from_explicit_graph, Arc
from shutil import copyfile
import os


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
            [Arc('PR1', 'C1', 6), Arc('C1', 'PR1', 6),
             Arc('C1', 'alt.atheism', 10), Arc('alt.atheism', 'C1', 10),
             Arc('C1', 'C2', 16), Arc('C2', 'C1', 16),
             Arc('C1', 'C5', 17), Arc('C5', 'C1', 17),
             Arc('C1', 'talk.religion.misc', 11), Arc('talk.religion.misc', 'C1', 11),
             Arc('C2', 'comp.graphics', 11), Arc('comp.graphics', 'C2', 11),
             Arc('C2', 'comp.os.ms-windows.misc', 12), Arc('comp.os.ms-windows.misc', 'C2', 12),
             Arc('C2', 'C3', 16), Arc('C3', 'C2', 16),
             Arc('C2', 'C6', 18), Arc('C6', 'C2', 18),
             Arc('C2', 'C5', 17), Arc('C5', 'C2', 17),
             Arc('C3', 'comp.os.ms-windows.misc', 10), Arc('comp.os.ms-windows.misc', 'C3', 10),
             Arc('C3', 'comp.sys.ibm.pc.hardware', 13), Arc('comp.sys.ibm.pc.hardware', 'C3', 13),
             Arc('C3', 'C4', 16), Arc('C4', 'C3', 16),
             Arc('C3', 'C7', 16), Arc('C7', 'C3', 16),
             Arc('C3', 'C6', 18), Arc('C6', 'C3', 18),
             Arc('C4', 'comp.sys.ibm.pc.hardware', 12), Arc('comp.sys.ibm.pc.hardware', 'C4', 12),
             Arc('C4', 'comp.sys.mac.hardware', 11), Arc('comp.sys.mac.hardware', 'C4', 11),
             Arc('C4', 'PR2', 6), Arc('PR2', 'C4', 6),
             Arc('C4', 'comp.windows.x', 13), Arc('comp.windows.x', 'C4', 13),
             Arc('C4', 'misc.forsale', 11), Arc('misc.forsale', 'C4', 11),
             Arc('C4', 'C7', 17), Arc('C7', 'C4', 17),
             Arc('C7', 'rec.autos', 10), Arc('rec.autos', 'C7', 10),
             Arc('C7', 'C11', 18), Arc('C11', 'C7', 18),
             Arc('C7', 'C10', 17), Arc('C10', 'C7', 17),
             Arc('C7', 'C6', 18), Arc('C6', 'C7', 18),
             Arc('C6', 'C10', 18), Arc('C10', 'C6', 18),
             Arc('C6', 'C9', 17), Arc('C9', 'C6', 17),
             Arc('C6', 'C5', 16), Arc('C5', 'C6', 16),
             Arc('C5', 'C9', 18), Arc('C9', 'C5', 18),
             Arc('C5', 'C8', 17), Arc('C8', 'C5', 17),
             Arc('C5', 'talk.politics.guns', 13), Arc('talk.politics.guns', 'C5', 13),
             Arc('C5', 'talk.politics.mideast', 12), Arc('talk.politics.mideast', 'C5', 12),
             Arc('C5', 'talk.politics.misc', 11), Arc('talk.politics.misc', 'C5', 11),
             Arc('C8', 'soc.religion.christian', 13), Arc('soc.religion.christian', 'C8', 13),
             Arc('C8', 'PR4', 7), Arc('PR4', 'C8', 7),
             Arc('C8', 'sci.space', 12), Arc('sci.space', 'C8', 12),
             Arc('C8', 'sci.med', 10), Arc('sci.med', 'C8', 10),
             Arc('C8', 'C9', 17), Arc('C9', 'C8', 17),
             Arc('C9', 'sci.med', 11), Arc('sci.med', 'C9', 11),
             Arc('C9', 'sci.electronics', 11), Arc('sci.electronics', 'C9', 11),
             Arc('C9', 'C10', 18), Arc('C10', 'C9', 18),
             Arc('C10', 'sci.electronics', 13), Arc('sci.electronics', 'C10', 13),
             Arc('C10', 'sci.crypt', 12), Arc('sci.crypt', 'C10', 12),
             Arc('C10', 'C11', 16), Arc('C11', 'C10', 16),
             Arc('C11', 'sci.crypt', 10), Arc('sci.crypt', 'C11', 10),
             Arc('C11', 'rec.sport.hockey', 12), Arc('rec.sport.hockey', 'C11', 12),
             Arc('C11', 'PR3', 8), Arc('PR3', 'C11', 8),
             Arc('C11', 'rec.sport.baseball', 10), Arc('rec.sport.baseball', 'C11', 10),
             Arc('C11', 'rec.motorcycles', 11), Arc('C4', 'rec.motorcycles', 11)]
        )

    def heuristic_builder(self, target_position):
        already_discovered = [target_position]
        if self.cyclic_delivery_problem.hmap:
            self.cyclic_delivery_problem.hmap.clear()
        self.cyclic_delivery_problem.hmap.update({str(target_position): 0})  # la distanza fino a se stesso è 0
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

    def path_searcher_with_df_branch_and_bound(self, target_position):
        self.heuristic_builder(target_position)
        self.cyclic_delivery_problem.start = self.current_position
        bound = 90          # 90 stima lunghezza massima percorso
        path_searcher = DF_branch_and_bound(self.cyclic_delivery_problem, bound)
        found_path = path_searcher.search()
        if found_path is not None:
            self.current_position = target_position
        return found_path

    def print_current_position(self):
        print('Agente in posizione: ', self.current_position)

    def go_back_to_resting_point(self):
        print('Cerco resting point più vicino...')
        closes_resting_point_node = ''
        closes_resting_point_distance = 100
        for i in range(1, 5, 1):
            resting_node = 'PR' + str(i)
            self.heuristic_builder(resting_node)
            if self.cyclic_delivery_problem.hmap[self.current_position] < closes_resting_point_distance:
                closes_resting_point_node = resting_node
                closes_resting_point_distance = self.cyclic_delivery_problem.hmap[self.current_position]
        print('il resting point più vicino è: ', closes_resting_point_node)
        print('Calcolo del percorso verso il resting point in corso..')
        path_found = self.path_searcher_with_df_branch_and_bound(closes_resting_point_node)
        if path_found is not None:
            print(path_found)
            self.print_current_position()
        else:
            print('nessun percorso disponibile')

    def archive_document(self, category, file_name):
        self.print_current_position()
        print('Calcolo del percorso in corso..')
        path_found = self.path_searcher_with_df_branch_and_bound(category)
        if path_found is not None:
            print(path_found)
            self.print_current_position()
            try:
                destination = './Archive/' + category
                if not os.path.exists(destination):
                    os.makedirs(destination)
                destination += '/' + file_name
                copyfile(file_name, destination)
            except:
                print('Impossibile archiviare documento. ')
            else:
                print('documento archiviato correttamente.')
                self.go_back_to_resting_point()






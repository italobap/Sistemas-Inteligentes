import time
from vs.constants import VS
from a_star import AStar

class OnlineDFS:
    def __init__(self, explorer_name, cost_line=1.0, cost_diag=1.5):
        self.explorer = explorer_name
        self.untried = {}
        self.unbacktracked = []
        self.visited = []

        self.go_back_mode = False
        self.go_back_path = []

        self.cost_line = cost_line # the cost to move one step in the horizontal or vertical
        self.cost_diag = cost_diag # the cost to move one step in any diagonal

        self.incr =  {              # the increments for each walk action
            0: (0, -1),             #  u: Up
            1: (1, -1),             # ur: Upper right diagonal
            2: (1, 0),              #  r: Right
            3: (1, 1),              # dr: Down right diagonal
            4: (0, 1),              #  d: Down
            5: (-1, 1),             # dl: Down left left diagonal
            6: (-1, 0),             #  l: Left
            7: (-1, -1)             # ul: Up left diagonal
        }

        self.inverse_inc = {
            (0, -1): 0,             #  u: Up
            (1, -1): 1,             # ur: Upper right diagonal
            (1, 0): 2,              #  r: Right
            (1, 1): 3,              # dr: Down right diagonal
            (0, 1): 4,              #  d: Down
            (-1, 1): 5,             # dl: Down left left diagonal
            (-1, 0): 6,             #  l: Left
            (-1, -1): 7             # ul: Up left diagonal
        }

        if(explorer_name == 'EXPL_1'):      # up-right
            #self.priority_order = [0, 1, 2, 7, 3, 6, 5, 4]
            # tecnica priorizando ortogonais
            self.priority_order = [0, 2, 4, 6, 1, 3, 5, 7]
        elif(explorer_name == 'EXPL_2'):    # down-right
            #self.priority_order = [2, 3, 4, 1, 5, 0, 7, 6]
            # tecnica priorizando ortogonais
            ##self.priority_order = [2, 4, 6, 0, 3, 5, 7, 1]
            self.priority_order = [4, 2, 0, 6, 3, 1, 7, 5]  
        elif(explorer_name == 'EXPL_3'):    # down-left
            #self.priority_order = [4, 5, 6, 7, 3, 2, 1, 0]
            # tecnica priorizando ortogonais
            ##self.priority_order = [4, 6, 0, 2, 5, 7, 1, 3]
            self.priority_order = [4, 6, 2, 0, 5, 7, 1, 3]  
        else:                               # up-left
            #self.priority_order = [6, 7, 0, 5, 1, 4, 3, 2]
            # tecnica priorizando ortogonais
            ##self.priority_order = [6, 0, 2, 4, 7, 1, 3, 5]
            self.priority_order = [0, 6, 4, 2, 7, 1, 3, 5]

        self.print = False
    
    def get_next_position(self, coord, map, actions_res):
        #print(f"{self.explorer}")
        #print(f"{coord}")
        
        #print(f"{actions_res}")
        #print(f"{self.priority_order}")
        # vou ordernar o movimento pelas prioridades e considerar untried apenas os possíveis
        if self.go_back_mode:
            if len(self.go_back_path) > 0:
                go_back_to = self.go_back_path.pop()
                #print(f"{self.explorer} at {coord}: is going back to {coord[0] + go_back_to[0]}, {coord[1] + go_back_to[1]}")
                return go_back_to
            else:
                self.go_back_mode = False




        if coord not in self.visited:
            self.visited.append(coord)

        untried = []
        
        if coord in self.untried:
            untried = self.untried[coord]

            for i in untried[:]:
                if (coord[0] + self.incr[i][0], coord[1] + self.incr[i][1]) in self.visited and i in untried:
                    untried.remove(i)
        else:
            for i in self.priority_order:
                if actions_res[i] == VS.CLEAR and (coord[0] + self.incr[i][0], coord[1] + self.incr[i][1]) not in self.visited:
                    untried.append(i)
        
        if len(self.unbacktracked) > 0 :
            pos_bef = self.unbacktracked[-1]
            diff = (pos_bef[0] - coord[0], pos_bef[1] - coord[1])
            #já explorou a forma como entrou na celula, não precisa testar o movimento inverso
            inverse_before_move = self.inverse_inc[diff]
            if inverse_before_move in untried:
                untried.remove(inverse_before_move)

        self.untried[coord] = untried

        if len(untried) > 0:
            # pega o primeiro do untried como decisão de movimento e tira da lista
            next_move = untried.pop(0)    
            # print(f"{self.untried[coord]}")
            self.unbacktracked.append(coord)
        else:
            #precisa voltar

            # se vai voltar, talvez seja o caso de ver até onde precisa voltar e ver a melhor forma de voltar
            go_back_to = None
            counter = 0
            for pos in reversed(self.unbacktracked):
                counter += 1
                if pos in self.untried:
                    pos_untried = self.untried[pos]
                    for i in pos_untried[:]: 
                        if (pos[0] + self.incr[i][0], pos[1] + self.incr[i][1]) in self.visited and i in pos_untried:
                            pos_untried.remove(i)
                    self.untried[pos] = pos_untried

                    if len(pos_untried) > 0:
                        #print(f"{pos_untried}")
                        go_back_to = pos
                        #print(f"{self.explorer} must go back to ({go_back_to})")
                        break
                        #tem que voltar até essa posição

            if go_back_to is not None and counter > 1:
                a_star = AStar(map, self.cost_line, self.cost_diag)
                self.go_back_path, _ = a_star.search(coord, go_back_to)

                #entra em modo de go_back
                self.go_back_mode = True

                #tirar do unbacktracked
                while self.unbacktracked:
                    item = self.unbacktracked.pop()
                    if item == go_back_to:
                        break
                
                next_pos = self.go_back_path.pop()
                #print(f"{self.explorer} at {coord}: is going back to ({coord[0] + next_pos[0]}, {coord[1] + next_pos[1]})")
                return next_pos            
            else:
                #volta um passo
                pos_bef = self.unbacktracked.pop()
                diff = (pos_bef[0] - coord[0], pos_bef[1] - coord[1])
                next_move = self.inverse_inc[diff]
                #print(f"{self.explorer} at {coord}: is going back to {pos_bef}")



        
        #print(f"{self.visited}")
        #print(f"{self.unbacktracked[-20:]}")
        #montar a ordem 

        # time.sleep(0.3)
        return self.incr[next_move]

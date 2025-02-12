import heapq
from collections import deque
from vs.constants import VS
from map import Map

class AStar:
    def __init__(self, map, cost_line=1.0, cost_diag=1.5):
        self.map = map             # an instance of the class Map
        self.cost_line = cost_line # the cost to move one step in the horizontal or vertical
        self.cost_diag = cost_diag # the cost to move one step in any diagonal
        self.tlim = float('inf')    # when the walk time reach this threshold, the plan is aborted
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
    
    
    def heuristic(self, start, goal):
        # como pode se mover na diagonal, usa heuristica euclidiana, usando sem raiz pois tem menor custo computacional, mas mantem ordem de prioridade semelhante
        return (start[0] - goal[0])**2 + (start[1] - goal[1])**2
    
    def trace_path(self, came_from, current):
        path = []
        while current in came_from:
            prev = came_from[current]
            path.append((current[0] - prev[0], current[1] - prev[1])) # só os incrementos
            current = prev
        #path.reverse()
        return path    


    def search(self, start, goal, tlim=float('inf')):
        """ this method performs a a-star search.
            @param start the initial position
            @param goal  the goal position
            @returns     a plan (a list of actions defined as increments in x and y; and the time to execute the reverse plan
                         for instance: [(-1, 0), (-1, 1), (1, 1), (1, 0)] walk +1 in the x position, walk +1 in x and +1 in the y;  so on
                         In case of fail, it returns:
                         [], -1: no plan because the time limit was reached
                         [],  0: no path found between start and goal position
                         plan, time: a plan with the time required to execute (only walk actions)"""
        
        self.tlim = tlim
        open_list = [] # priority queue
        heapq.heappush(open_list, (0, start))  # (custo F(n), posição)

        came_from = {} # pra reconstruir o caminho
        came_from_cost = {}
        g_score = {start: 0} # custo acumulado real até cada posição

        while open_list:
            #print(f"open list: {open_list}")
            f_score, current_pos = heapq.heappop(open_list)

            # se deu o tempo
            if g_score[current_pos] > tlim:
                return [], -1
            
            # se chegou no objetivo
            if current_pos == goal:
                path = self.trace_path(came_from, current_pos)
                #print(f"{path}")
                total_time = g_score[current_pos] #TODO arrumar isso
                return path, total_time
            
            #expansão dos vizinhos
            for direction, (dx, dy) in self.incr.items():
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)

                if not self.map.in_map(neighbor):
                    continue

                difficulty = self.map.get_difficulty(neighbor)
                move_cost = self.cost_diag if dx != 0 and dy != 0 else self.cost_line
                g_neighbor = g_score[current_pos] + (move_cost * difficulty)

                # se for um caminho melhor, atualiza
                if neighbor not in g_score or g_neighbor < g_score[neighbor]:
                    g_score[neighbor] = g_neighbor
                    f_score = g_neighbor + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current_pos
            
        return [], 0
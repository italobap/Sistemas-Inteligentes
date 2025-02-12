import sys
import os
import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from online_dfs import OnlineDFS
from a_star import AStar


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0


class Explorer(AbstAgent):
    MAX_DIFFICULTY = 3  # Maximum difficulty for accessible cells

    def __init__(self, env, config_file, resc):
        """
        Explorer agent constructor.
        @param env: Reference to the environment.
        @param config_file: Absolute path to the agent's config file.
        @param resc: Reference to the master Rescuer agent.
        """
        super().__init__(env, config_file)
        self.walk_stack = Stack()
        self.walk_time = 0
        self.set_state(VS.ACTIVE)  # Starts in active state
        self.resc = resc  # Reference to the Rescuer agent
        self.x, self.y = 0, 0  # Relative position to the origin (0,0)
        self.map = Map()
        self.victims = {}  # Stores victim information
        self.visited = set()  # Tracks visited cells

        self.go_back_mode = False
        self.go_back_path = []

        # Add the starting position (base) to the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y))
        self.online_dfs = OnlineDFS(self.NAME, self.COST_LINE, self.COST_DIAG)

    def get_next_position(self):
        """ Gets the next position that can be explored (no wall and inside the grid)
        """
        # check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        #if self.NAME == "EXPL_1": # or self.NAME == "EXPL_2" : 
        return self.online_dfs.get_next_position((self.x, self.y), self.map, obstacles)
    
    # def get_next_position(self):
    #     """ Selects the next position to explore, prioritizing unexplored areas. """
    #     obstacles = self.check_walls_and_lim()

    #     # Get possible moves (CLEAR cells)
    #     possible_moves = [
    #         (direction, dx, dy)
    #         for direction, (dx, dy) in Explorer.AC_INCR.items()
    #         if obstacles[direction] == VS.CLEAR
    #     ]

    #     random.shuffle(possible_moves)  # Maintain randomness

    #     # Prioritize unexplored cells
    #     for _, dx, dy in possible_moves:
    #         next_pos = (self.x + dx, self.y + dy)
    #         if next_pos not in self.visited:
    #             return dx, dy

    #     # Fallback: Choose any valid move if all are explored
    #     if possible_moves:
    #         _, dx, dy = random.choice(possible_moves)
    #         return dx, dy

    #     return 0, 0  # Default (should rarely occur)

    # def adjust_difficulty(self):
    #     """ Adjusts MAX_DIFFICULTY dynamically based on explored area size. """
    #     explored_cells = len(self.map.data)
    #     if explored_cells > 50:
    #         Explorer.MAX_DIFFICULTY = 2
    #     elif explored_cells > 100:
    #         Explorer.MAX_DIFFICULTY = 3

    def explore(self):
        #get an random increment for x and y
        dx, dy = self.get_next_position()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")
        
        if result == VS.EXECUTED:

            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            #update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                # if VS.DEBUG : print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                # if VS.DEBUG : print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if(dx == 0 or dy == 0):
                difficulty = difficulty / self.COST_LINE
            else: 
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #if VS.DEBUG and self.NAME == "EXPL_1" : print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")
        return
    # def explore(self):
    #     """ Explores the environment by moving to a new position. """
    #     self.adjust_difficulty()
    #     dx, dy = self.get_next_position()

    #     rtime_bef = self.get_rtime()
    #     result = self.walk(dx, dy)
    #     rtime_aft = self.get_rtime()

    #     if result == VS.BUMPED:
    #         self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

    #     elif result == VS.EXECUTED:
    #         self.walk_stack.push((dx, dy))
    #         self.x += dx
    #         self.y += dy
    #         self.walk_time += (rtime_bef - rtime_aft)
    #         self.visited.add((self.x, self.y))

    #         # Check for victims
    #         seq = self.check_for_victim()
    #         if seq != VS.NO_VICTIM:
    #             vs = self.read_vital_signals()
    #             self.victims[vs[0]] = ((self.x, self.y), vs)

    #         # Calculate cell difficulty
    #         difficulty = (rtime_bef - rtime_aft)
    #         difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG
    #         self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

    def come_back(self):
        next_move = (0,0)
        if self.go_back_mode:
            if len(self.go_back_path) > 0:
                next_move = self.go_back_path.pop()
                #print(f"{self.NAME} at ({self.x}, {self.y}): is going back to {self.x + next_move[0]}, {self.y + next_move[1]}")
            else:
                self.go_back_mode = False
        else:
            print(f"{self.NAME} GOING BACK TO BASE")
            a_star = AStar(self.map, self.COST_LINE, self.COST_DIAG)
            self.go_back_path, total_time = a_star.search((self.x, self.y), (0,0), self.get_rtime())
            # print(f"{self.go_back_path}")
            # print(f"{total_time}")

            self.go_back_mode = True
                
            next_move = self.go_back_path.pop()
            #print(f"{self.NAME} at ({self.x}, {self.y}): is going back to ({self.x + next_move[0]}, {self.y + next_move[1]})") 

        #dx, dy = self.walk_stack.pop()
        dx = next_move[0]
        dy = next_move[1]
        #dx = dx * -1
        #dy = dy * -1
        result = self.walk(dx, dy)

        if result == VS.BUMPED:
            print(f"BUMPED")
            return
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    # def come_back(self):
    #     """ Retraces steps to return to the base. """
    #     if not self.walk_stack.is_empty():
    #         dx, dy = self.walk_stack.pop()
    #         dx, dy = -dx, -dy
    #         result = self.walk(dx, dy)

    #         if result == VS.EXECUTED:
    #             self.x += dx
    #             self.y += dy

    def deliberate(self) -> bool:
        
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 4 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ
        
        # keeps exploring while there is enough time
        if self.walk_time < (self.get_rtime() - time_tolerance):
        #if 500 < (self.get_rtime() - time_tolerance):
            self.explore()
            return True
        else:
            a_star = AStar(self.map, self.COST_LINE, self.COST_DIAG)
            _, expected_time  = a_star.search((self.x, self.y), (0,0))
            #DEBUG print
            #print(f"{self.NAME} - et: {expected_time}, rt: {self.get_rtime() - time_tolerance}")

            if expected_time != -1 and expected_time != 0 and expected_time < (self.get_rtime() - time_tolerance):
                self.explore()
                return True
        
        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True
    
    # def deliberate(self) -> bool:
    #     """ Main decision loop for the Explorer. """
    #     time_tolerance = 2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

    #     # Emergency return if time is critically low
    #     if self.get_rtime() <= time_tolerance + (len(self.walk_stack.items) * self.COST_LINE):
    #         # Emergency return logic
    #         if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
    #             self.resc.sync_explorers(self.map, self.victims)
    #             return False
    #         self.come_back()
    #         return True

    #     # Continue exploring if sufficient time is available
    #     if self.walk_time < (self.get_rtime() - time_tolerance):
    #         self.explore()
    #         return True

    #     # Default return logic
    #     if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
    #         self.resc.sync_explorers(self.map, self.victims)
    #         return False
    #     self.come_back()
    #     return True


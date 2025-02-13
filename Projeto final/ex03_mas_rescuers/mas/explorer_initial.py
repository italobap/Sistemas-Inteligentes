import sys
import os
import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map


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
    MAX_DIFFICULTY = 1  # Maximum difficulty for accessible cells

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

        # Add the starting position (base) to the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y))

    def get_next_position(self):
        """ Selects the next position to explore, prioritizing unexplored areas. """
        obstacles = self.check_walls_and_lim()

        # Get possible moves (CLEAR cells)
        possible_moves = [
            (direction, dx, dy)
            for direction, (dx, dy) in Explorer.AC_INCR.items()
            if obstacles[direction] == VS.CLEAR
        ]

        random.shuffle(possible_moves)  # Maintain randomness

        # Prioritize unexplored cells
        for _, dx, dy in possible_moves:
            next_pos = (self.x + dx, self.y + dy)
            if next_pos not in self.visited:
                return dx, dy

        # Fallback: Choose any valid move if all are explored
        if possible_moves:
            _, dx, dy = random.choice(possible_moves)
            return dx, dy

        return 0, 0  # Default (should rarely occur)

    def adjust_difficulty(self):
        """ Adjusts MAX_DIFFICULTY dynamically based on explored area size. """
        explored_cells = len(self.map.data)
        if explored_cells > 50:
            Explorer.MAX_DIFFICULTY = 2
        elif explored_cells > 100:
            Explorer.MAX_DIFFICULTY = 3

    def explore(self):
        """ Explores the environment by moving to a new position. """
        self.adjust_difficulty()
        dx, dy = self.get_next_position()

        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        elif result == VS.EXECUTED:
            self.walk_stack.push((dx, dy))
            self.x += dx
            self.y += dy
            self.walk_time += (rtime_bef - rtime_aft)
            self.visited.add((self.x, self.y))

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)

            # Calculate cell difficulty
            difficulty = (rtime_bef - rtime_aft)
            difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

    def come_back(self):
        """ Retraces steps to return to the base. """
        if not self.walk_stack.is_empty():
            dx, dy = self.walk_stack.pop()
            dx, dy = -dx, -dy
            result = self.walk(dx, dy)

            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy

    def deliberate(self) -> bool:
        """ Main decision loop for the Explorer. """
        time_tolerance = 2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # Emergency return if time is critically low
        if self.get_rtime() <= time_tolerance + (len(self.walk_stack.items) * self.COST_LINE):
            # Emergency return logic
            if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
                self.resc.sync_explorers(self.map, self.victims)
                return False
            self.come_back()
            return True

        # Continue exploring if sufficient time is available
        if self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # Default return logic
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            self.resc.sync_explorers(self.map, self.victims)
            return False
        self.come_back()
        return True


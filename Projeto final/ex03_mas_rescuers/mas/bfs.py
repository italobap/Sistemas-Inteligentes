from collections import deque
from vs.constants import VS
from heapq import heappush, heappop
from map import Map

class BFS:
    def __init__(self, map, cost_line=1.0, cost_diag=1.5):
        self.map = map             # Instance of the Map class
        self.cost_line = cost_line # Cost for horizontal or vertical moves
        self.cost_diag = cost_diag # Cost for diagonal moves
        self.incr = {              # Movement increments for each action
            0: (0, -1),            # Up
            1: (1, -1),            # Upper right diagonal
            2: (1, 0),             # Right
            3: (1, 1),             # Down right diagonal
            4: (0, 1),             # Down
            5: (-1, 1),            # Down left diagonal
            6: (-1, 0),            # Left
            7: (-1, -1)            # Upper left diagonal
        }

    def get_possible_actions(self, pos):
        """Find valid moves from the current position."""
        x, y = pos
        actions = []

        if self.map.in_map(pos):
            for direction, (dx, dy) in self.incr.items():
                neighbor = (x + dx, y + dy)
                if self.map.in_map(neighbor):
                    cell_status = self.map.get_actions_results(pos)[direction]
                    if cell_status == VS.CLEAR:
                        actions.append((dx, dy))
        
        return actions

    def get_unexplored_neighbors(self, pos):
        """Find neighbors with unexplored directions (`VS.UNK`)."""
        x, y = pos
        unexplored = []

        if self.map.in_map(pos):
            for direction, (dx, dy) in self.incr.items():
                neighbor = (x + dx, y + dy)
                if self.map.in_map(neighbor):
                    cell_status = self.map.get_actions_results(pos)[direction]
                    if cell_status == VS.UNK:
                        unexplored.append((dx, dy))

        return unexplored

    def heuristic(self, current, goal):
        """Calculate the Manhattan distance heuristic for A*."""
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def search(self, start, goal, method="a_star", tlim=float('inf')):
        """
        Perform pathfinding using Uniform Cost Search (UCS) or A*.

        Args:
            start: Start position as (x, y).
            goal: Goal position as (x, y).
            method: "ucs" for Uniform Cost Search, "a_star" for A*.
            tlim: Time limit for the search.

        Returns:
            plan: List of actions (increments in x and y).
            total_cost: Cost of the plan, or -1 if the time limit is exceeded.
        """
        if not self.map.in_map(start) or not self.map.in_map(goal):
            raise ValueError("Start or goal position is outside the map boundaries.")

        if start == goal:
            return [], 0

        frontier = []  # Priority queue for UCS or A*
        heappush(frontier, (0, start, []))  # (cost, position, plan)

        visited = set()

        while frontier:
            cost, current_pos, plan = heappop(frontier)

            if current_pos in visited:
                continue
            
            visited.add(current_pos)

            if current_pos == goal:
                if cost > tlim:
                    return [], -1  # Time limit exceeded
                return plan, cost

            for action in self.get_possible_actions(current_pos):
                neighbor = (current_pos[0] + action[0], current_pos[1] + action[1])

                if neighbor not in visited:
                    difficulty = self.map.get_difficulty(neighbor)
                    step_cost = self.cost_line if action[0] == 0 or action[1] == 0 else self.cost_diag
                    total_cost = cost + step_cost * difficulty

                    if total_cost > tlim:
                        continue

                    new_plan = plan + [action]

                    priority = total_cost
                    if method == "a_star":
                        priority += self.heuristic(neighbor, goal)

                    heappush(frontier, (priority, neighbor, new_plan))

        return [], 0  # No path found

# Example usage
if __name__ == '__main__':
    map = Map()
    map.data = {
        (0, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END]),
        (1, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END]),
        (2, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.WALL, VS.WALL, VS.CLEAR, VS.CLEAR, VS.END]),
        (3, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.END, VS.END, VS.WALL, VS.WALL, VS.CLEAR, VS.END]),
        (0, 1): (1, 1, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END, VS.END]),
        (1, 1): (1, 2, [VS.CLEAR, VS.CLEAR, VS.WALL, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR]),
        (0, 2): (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END, VS.END, VS.END, VS.END]),
        (1, 2): (1, VS.NO_VICTIM, [VS.CLEAR, VS.WALL, VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (2, 2): (1, VS.NO_VICTIM, [VS.WALL, VS.WALL, VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (3, 2): (1, 3, [VS.WALL, VS.END, VS.END, VS.END, VS.END, VS.END, VS.CLEAR, VS.WALL]),
    }
    map.draw()

    start = (3, 0)
    goal = (3, 2)

    pathfinder = BFS(map)

    # Using Uniform Cost Search
    plan_ucs, cost_ucs = pathfinder.search(start, goal, method="ucs")
    print(f"UCS Plan: {plan_ucs}, Cost: {cost_ucs}")

    # Using A*
    plan_a_star, cost_a_star = pathfinder.search(start, goal, method="a_star")
    print(f"A* Plan: {plan_a_star}, Cost: {cost_a_star}")

    # Find unexplored neighbors
    unexplored = pathfinder.get_unexplored_neighbors((0, 0))
    print(f"Unexplored Neighbors: {unexplored}")

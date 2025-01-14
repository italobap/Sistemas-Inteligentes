##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import numpy as np
from sklearn.cluster import KMeans
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from heapq import heappush, heappop
from map import Map


class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[]):
        super().__init__(env, config_file)
        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map()
        self.victims = {}
        self.plan = []
        self.plan_rtime = self.TLIM
        self.x = 0
        self.y = 0
        self.clusters = clusters
        self.sequences = clusters
        self.rescuers = [self]  # List of all rescuers, including self
        self.set_state(VS.IDLE)

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1
  
            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method 
         
        

    def predict_severity_and_class(self):
        """Assign random severity values and classes to victims."""
        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 100.0)
            severity_class = random.randint(1, 4)
            values[1].extend([severity_value, severity_class])

    def cluster_victims(self, n_clusters=4):
        """Cluster victims using K-Means."""
        if not self.victims:
            return []

        victim_positions = np.array([values[0] for values in self.victims.values()])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(victim_positions)

        clusters = [{} for _ in range(n_clusters)]
        for label, (key, values) in zip(labels, self.victims.items()):
            clusters[label][key] = values

        return clusters

    def sequencing(self):
        """Optimize rescue order using a Genetic Algorithm."""
        def fitness(sequence):
            total_distance = 0
            severity_score = 0
            start = (0, 0)
            for vic_id in sequence:
                position = sequence[vic_id][0]
                severity = sequence[vic_id][1][-1]  # Assume severity is the last element
                total_distance += math.dist(start, position)
                severity_score += severity
                start = position
            return total_distance - severity_score  # Lower is better

        def mutate(sequence):
            idx1, idx2 = random.sample(range(len(sequence)), 2)
            sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]

        def crossover(seq1, seq2):
            mid = len(seq1) // 2
            return seq1[:mid] + seq2[mid:]

        def genetic_algorithm(victims, n_generations=100, population_size=10):
            population = [list(victims.items()) for _ in range(population_size)]
            for _ in range(n_generations):
                population.sort(key=lambda seq: fitness(dict(seq)))
                next_population = population[:population_size // 2]
                for i in range(len(next_population) // 2):
                    child = crossover(next_population[i], next_population[-i - 1])
                    mutate(child)
                    next_population.append(child)
                population = next_population
            return dict(population[0])

        self.sequences = [genetic_algorithm(cluster) for cluster in self.clusters]

    def planner(self):
        """Calculate paths between victims using A* while respecting TLIM."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def a_star(start, goal):
            open_set = []
            heappush(open_set, (0, start, []))
            g_score = {start: 0}
            while open_set:
                _, current, path = heappop(open_set)
                if current == goal:
                    return path
                for dx, dy in self.map.get_possible_actions(current):
                    neighbor = (current[0] + dx, current[1] + dy)
                    difficulty = self.map.get_difficulty(neighbor)
                    if difficulty is None:
                        continue  # Skip neighbors not in the map

                    tentative_g_score = g_score[current] + difficulty
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heappush(open_set, (f_score, neighbor, path + [(dx, dy)]))
            return []

        if not self.sequences:
            print(f"{self.NAME}: No sequences available for planning.")
            return

        total_time = 0
        sequence = self.sequences[0]
        start = (0, 0)
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            path = a_star(start, goal)
            travel_time = len(path) * self.COST_LINE
            first_aid_time = self.COST_FIRST_AID
            total_time += travel_time + first_aid_time

            if total_time > self.TLIM:
                print(f"{self.NAME}: Time limit exceeded during planning. Stopping at victim {vic_id}.")
                break

            self.plan += path
            start = goal

        # Plan return to base
        return_path = a_star(start, (0, 0))
        if total_time + len(return_path) * self.COST_LINE <= self.TLIM:
            self.plan += return_path
        else:
            print(f"{self.NAME}: Not enough time to return to base. Plan ends at {start}.")


    def deliberate(self) -> bool:
        """Execute the next action in the plan, respecting TLIM."""
        if not self.plan:
            print(f"{self.NAME}: Finished plan.")
            return False

        remaining_time = self.get_rtime()
        next_step_cost = self.COST_LINE if (self.plan[0][0] == 0 or self.plan[0][1] == 0) else self.COST_DIAG

        if remaining_time < next_step_cost:
            print(f"{self.NAME}: Not enough time to execute next step. Stopping.")
            return False

        dx, dy = self.plan.pop(0)
        walked = self.walk(dx, dy)
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM and remaining_time >= self.COST_FIRST_AID:
                    self.first_aid()
        else:
            print(f"{self.NAME}: Walk failed at ({self.x}, {self.y}).")
        return True


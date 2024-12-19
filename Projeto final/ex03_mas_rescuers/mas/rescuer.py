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

    def sync_explorers(self, explorer_map, victims):
        """Synchronize with explorers and assign clusters to all rescuers."""
        self.received_maps += 1
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME}: All maps received from explorers.")
            self.predict_severity_and_class()

            # Cluster victims using K-Means
            all_clusters = self.cluster_victims(n_clusters=4)

            # Assign clusters to rescuers
            for i, rescuer in enumerate(self.rescuers):
                if i < len(all_clusters):
                    rescuer.clusters = [all_clusters[i]]
                else:
                    rescuer.clusters = []

            # Perform planning for each rescuer
            for rescuer in self.rescuers:
                rescuer.sequences = rescuer.clusters
                rescuer.sequencing()
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

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
        """Calculate paths between victims using A*."""
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

        sequence = self.sequences[0]
        start = (0, 0)
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan = a_star(start, goal)
            self.plan += plan
            start = goal

        # Plan return to base
        plan = a_star(start, (0, 0))
        self.plan += plan

    def deliberate(self) -> bool:
        """Execute the next action in the plan."""
        if not self.plan:
            print(f"{self.NAME}: Finished plan.")
            return False

        dx, dy = self.plan.pop(0)
        walked = self.walk(dx, dy)
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
        else:
            print(f"{self.NAME}: Walk failed at ({self.x}, {self.y}).")
        return True

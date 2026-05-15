import time
from parsing_data import *
from fitness_calculation import *
from crossing import *
from mutation import *
from greedy_ts import *
import math

mutation_rate = 0.25
chromosomes_amount = 20
preserved_precentage = 5
proportion = 0.25
generations = 1500
max_time = 10


preserved_chromosomes = math.ceil(chromosomes_amount * (preserved_precentage / 100))
max_new_chromosomes = chromosomes_amount - preserved_chromosomes

chromosomes = list()
fitness_list = list()


def sorting_chromosomes(chromosomes, fitness_list):
    sorted_data = sorted(zip(chromosomes, fitness_list), key=lambda x: x[1])
    return zip(*sorted_data)


def initial_generation(tasks, chromosomes_amount, chromosomes, fitness_list):
    chromosomes.append(greedy_schedule(tasks, machines))
    for _ in range(chromosomes_amount - 1):
        chromosomes.append([random.randint(0, machines - 1) for task in tasks])

    for k in range(chromosomes_amount):
        fitness_list.append(fitness(machines, chromosomes[k], tasks))
    chromosomes, fitness_list = sorting_chromosomes(chromosomes, fitness_list)

    return chromosomes, fitness_list


def evolution(chromosomes, fitness_list, mutation_rate, generation):
    best_chromosomes = chromosomes[:preserved_chromosomes]
    new_chromosomes = list()
    current_mutation_rate = mutation_rate

    if (max_new_chromosomes % 2 == 1):
        for k in range(max_new_chromosomes // 2 + 1):
            index1 = random.randint(0, chromosomes_amount - 1)
            index2 = random.randint(0, chromosomes_amount - 1)
            new_chromosome1, new_chromosome2 = crossing(chromosomes[index1], chromosomes[index2], proportion)
            new_chromosomes.extend([new_chromosome1, new_chromosome2])
        new_chromosomes.pop()
    else:
        for k in range(max_new_chromosomes // 2):
            index1 = random.randint(0, chromosomes_amount - 1)
            index2 = random.randint(0, chromosomes_amount - 1)
            new_chromosome1, new_chromosome2 = crossing(chromosomes[index1], chromosomes[index2], proportion)
            new_chromosomes.extend([new_chromosome1, new_chromosome2])

    chromosomes = list()
    new_chromosomes = [mutation(current_mutation_rate, chromosome, machines, random.randint(1, machines)) for chromosome in new_chromosomes]
    chromosomes += best_chromosomes
    chromosomes += new_chromosomes
    fitness_list = list()
    for k in range(chromosomes_amount):
        fitness_list.append(fitness(machines, chromosomes[k], tasks))
    chromosomes, fitness_list = sorting_chromosomes(chromosomes, fitness_list)

    return chromosomes, fitness_list


machines, task_amount, tasks = parse()

s = 0
for k in range(len(tasks)):
    s += tasks[k]
lb = s / machines

t = time.process_time()

chromosomes, fitness_list = initial_generation(tasks, chromosomes_amount, chromosomes, fitness_list)
best_chromosome = [chromosomes[0], fitness(machines, chromosomes[0], tasks), 1]

for k in range(2, generations + 1):
    chromosomes, fitness_list = evolution(chromosomes, fitness_list, mutation_rate, k)
    if min(fitness_list) < best_chromosome[1]:
        best_chromosome[0] = chromosomes[0]
        best_chromosome[1] = fitness_list[0]
        best_chromosome[2] = k
    if time.process_time() - t > max_time:
        print(best_chromosome[1])
        print(lb)
        exit(0)

t = time.process_time() - t

print(t)
print(chromosomes)
print(fitness_list)

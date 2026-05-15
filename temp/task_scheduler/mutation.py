import random


def mutation(mutation_rate, chromosome, machines, numeric_mutation_range=1):
    current_mutation_rate = mutation_rate

    mutation_position = random.randint(0, len(chromosome) - 1)
    # mutation_indices = list(range(max(0, mutation_position - 3), min(len(chromosome), mutation_position + 4)))
    mutation_indices = [i for i in range(max(0, mutation_position - 3), min(len(chromosome), mutation_position + 4))]

    for index in mutation_indices:
        if random.random() < current_mutation_rate:
            mutation_strategy = random.choices(
                ['swap', 'random', 'increment', 'decrement'],
                weights=[0.2, 0.2, 0.3, 0.3])[0]

            if mutation_strategy == 'swap':
                swap_index = random.choice(mutation_indices)
                chromosome[index], chromosome[swap_index] = chromosome[swap_index], chromosome[index]
            elif mutation_strategy == 'random':
                chromosome[index] = random.randint(0, machines - 1)
            elif mutation_strategy == 'increment':
                chromosome[index] = (chromosome[index] + numeric_mutation_range) % machines
            elif mutation_strategy == 'decrement':
                chromosome[index] = (chromosome[index] - numeric_mutation_range) % machines

    return chromosome

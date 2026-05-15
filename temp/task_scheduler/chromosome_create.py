def create_chromosome(machines, num_tasks, task_list):
    chromosome = list()
    machine = 0
    for task in task_list:
        chromosome.append({'task': task, 'machine': (machine % machines) + 1})
        machine += 1

    return chromosome

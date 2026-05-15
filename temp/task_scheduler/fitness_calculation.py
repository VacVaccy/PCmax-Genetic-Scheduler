def fitness(num_machines, chromosome, tasks_list):
    time_lists = [0] * num_machines

    for task, processing_time in zip(chromosome, tasks_list):
        time_lists[task - 1] += processing_time

    Cmax = max(time_lists)
    return Cmax

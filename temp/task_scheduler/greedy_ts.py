def greedy_schedule(tasks, machines):
    task_order = list(range(len(tasks)))

    task_order.sort(key=lambda task: tasks[task], reverse=True)

    schedule = [None] * len(tasks)

    for task in task_order:
        machine_loads = [sum(1 for scheduled_task in schedule if scheduled_task == m) for m in range(machines)]
        min_load_machine = machine_loads.index(min(machine_loads))
        schedule[task] = min_load_machine

    return schedule

def parse():
    data = open("data.txt", "r")
    tasks = list()
    machines = int(data.readline().strip())
    task_amount = int(data.readline().strip())
    for item in data:
        tasks.append(int(item))
    data.close()
    return machines, task_amount, tasks

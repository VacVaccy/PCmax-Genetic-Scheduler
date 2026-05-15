import random

machines_min = 100
machines_max = 100
tasks_min = 800
tasks_max = 800


def generate_instance():
    array = list()
    array.append(random.randint(machines_min, machines_max))
    array.append(random.randint(tasks_min, tasks_max))
    for _ in range(array[1]):
        array.append(random.randint(1, 100))
    return array


def instance_to_file():
    data_list = generate_instance()
    data = open("data.txt", "w")
    for k in range(len(data_list)):
        data.write(f"{str(data_list[k])}\n")
    data.close()


instance_to_file()

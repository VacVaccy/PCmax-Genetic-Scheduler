def crossing(chromosome1, chromosome2, proportion):
    proportion = max(0.1, min(0.9, proportion))
    split_point = int(len(chromosome1) * proportion)
    new_chromosome1 = chromosome1[:split_point] + chromosome2[split_point:]
    new_chromosome2 = chromosome2[:split_point] + chromosome1[split_point:]

    return new_chromosome1, new_chromosome2

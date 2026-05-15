#include "utils.hpp"

// Init greedy approach for better start (LPT)
vector<Gene> greedy(int numMachines, vector<int>& tasks) {
    vector<int> taskOrder(tasks.size());
    vector<int> machinesLoad(numMachines, 0); // Total time per machine
    vector<Gene> chromosome;

    int minMachine = -1;

    iota(taskOrder.begin(), taskOrder.end(), 0);

    // Sorting by decreasing time
    sort(taskOrder.begin(), taskOrder.end(),
        [&](int a, int b) {
            return tasks[a] > tasks[b];
        }
    );

    for (int task : taskOrder) {
        // Find machine with min total time
        minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back({task, minMachine + 1});
        machinesLoad[minMachine] += tasks[task]; 
    }

    return chromosome;
}


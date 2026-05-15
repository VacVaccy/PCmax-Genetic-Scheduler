#include "utils.hpp"

pair<int, vector<int>> parseData(const string& filename) {
    ifstream file(filename);
    int numMachines, task_count;
    file >> numMachines >> task_count;

    vector<int> tasks(task_count);
    for (int i = 0; i < task_count; ++i) {
        file >> tasks[i];
    }
    return {numMachines, tasks};
}
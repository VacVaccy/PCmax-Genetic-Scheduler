#include "utils.hpp"

// // Applying mutations with strategies
// vector<Gene> mutation(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange) {
//     random_device rd;
//     mt19937 gen(rd());
//     vector<int> mutationIndexes;

//     double currentMutationRate = mutationRate;
//     uniform_int_distribution<> positionDist(0, chromosome.size() - 1);
//     uniform_real_distribution<> chanceDist(0.0, 1.0);
//     uniform_int_distribution<> machineDist(0, numMachines - 1);
//     vector<string> strategies = {"swap", "random", "increment", "decrement"}; // Strategies
//     vector<double> weights = {0.2, 0.2, 0.3, 0.3}; // Chance for strategy being used

//     int mutationPosition = positionDist(gen); // Center of mutation
//     int start = max(0, mutationPosition - 3); // Left bound
//     int end = min(static_cast<int>(chromosome.size()), mutationPosition + 4); // Right bound

//     for (int i = start; i < end; ++i) {
//         mutationIndexes.push_back(i);
//     }

//     for (int index : mutationIndexes) {
//         if (chanceDist(gen) < currentMutationRate) {
//             discrete_distribution<> strategyDist(weights.begin(), weights.end());
//             string mutationStrategy = strategies[strategyDist(gen)];

//             if (mutationStrategy == "swap") {
//                 uniform_int_distribution<> swapDist(0, mutationIndexes.size() - 1);
//                 int swapIndex = mutationIndexes[swapDist(gen)];
//                 swap(chromosome[index].machine, chromosome[swapIndex].machine);
//             } else if (mutationStrategy == "random") {
//                 chromosome[index].machine = machineDist(gen);
//             } else if (mutationStrategy == "increment") {
//                 chromosome[index].machine = (chromosome[index].machine + mutationRange) % numMachines;
//             } else if (mutationStrategy == "decrement") {
//                 chromosome[index].machine = (chromosome[index].machine - mutationRange + numMachines) % numMachines;
//             }
//         }
//     }

//     return chromosome;
// }


void mutation_inplace(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange) {
    random_device rd;
    mt19937 gen(rd());
    vector<int> mutationIndexes;

    double currentMutationRate = mutationRate;
    uniform_int_distribution<> positionDist(0, chromosome.size() - 1);
    uniform_real_distribution<> chanceDist(0.0, 1.0);
    uniform_int_distribution<> machineDist(1, numMachines);
    vector<string> strategies = {"swap", "random", "increment", "decrement"};
    vector<double> weights = {0.2, 0.2, 0.3, 0.3};

    int mutationPosition = positionDist(gen);
    int start = max(0, mutationPosition - 3);
    int end = min(static_cast<int>(chromosome.size()), mutationPosition + 4);

    for (int i = start; i < end; ++i) {
        mutationIndexes.push_back(i);
    }

    for (int index : mutationIndexes) {
        if (chanceDist(gen) < currentMutationRate) {
            discrete_distribution<> strategyDist(weights.begin(), weights.end());
            string mutationStrategy = strategies[strategyDist(gen)];

            if (mutationStrategy == "swap") {
                uniform_int_distribution<> swapDist(0, mutationIndexes.size() - 1);
                int swapIndex = mutationIndexes[swapDist(gen)];
                swap(chromosome[index].machine, chromosome[swapIndex].machine);
            } 
            else if (mutationStrategy == "random") {
                chromosome[index].machine = machineDist(gen);
            } 
            else if (mutationStrategy == "increment") {
                chromosome[index].machine = (chromosome[index].machine + mutationRange) % numMachines;
                if (chromosome[index].machine == 0) chromosome[index].machine = numMachines;
            } 
            else if (mutationStrategy == "decrement") {
                chromosome[index].machine = (chromosome[index].machine - mutationRange + numMachines) % numMachines;
                if (chromosome[index].machine == 0) chromosome[index].machine = numMachines;
            }
        }
    }
}
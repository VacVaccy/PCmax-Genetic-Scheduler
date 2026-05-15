#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <ctime>
#include <iostream>
#include <numeric>
#include <cmath>
#include <chrono>
#include <tuple>
#include <set>
#include <unordered_set>

using namespace std;

struct Gene {
    int task;
    int machine;

    Gene(int t, int m) : task(t), machine(m) {}
};

struct BestChromosome {
    vector<Gene> chromosome;
    int fitness;
    int generation;
};

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion, const vector<int>& tasks);
pair<int, vector<int>> parseData(const string& filename);
int fitnessCalculation(int machines, const vector<Gene>& chromosome, const vector<int>& tasks);
vector<Gene> greedy(int numMachines, vector<int>& tasks);
void mutation(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange, const vector<int>& tasks, double pressure);



// void mutation(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange, const vector<int>& tasks) {
//     random_device rd;
//     mt19937 gen(rd());
//     vector<int> mutationIndexes;

//     uniform_int_distribution<> positionDist(0, chromosome.size() - 1);
//     uniform_real_distribution<> chanceDist(0.0, 1.0);
//     uniform_int_distribution<> machineDist(0, numMachines - 1);
//     vector<string> strategies = {"swap", "random", "increment", "decrement"};
//     vector<double> weights = {0.2, 0.2, 0.3, 0.3};

//     int mutationPosition = positionDist(gen);
//     int start = max(0, mutationPosition - 3);
//     int end = min(static_cast<int>(chromosome.size()), mutationPosition + 4);

//     for (int i = start; i < end; ++i) {
//         mutationIndexes.push_back(i);
//     }

//     for (int index : mutationIndexes) {
//         if (chanceDist(gen) < mutationRate) {
//             discrete_distribution<> strategyDist(weights.begin(), weights.end());
//             string mutationStrategy = strategies[strategyDist(gen)];

//             if (mutationStrategy == "swap") {
//                 uniform_int_distribution<> swapDist(0, mutationIndexes.size() - 1);
//                 int swapIndex = mutationIndexes[swapDist(gen)];
//                 swap(chromosome[index].machine, chromosome[swapIndex].machine);
//             } 
//             else if (mutationStrategy == "random") {
//                 chromosome[index].machine = machineDist(gen);
//             } 
//             else if (mutationStrategy == "increment") {
//                 chromosome[index].machine = (chromosome[index].machine + mutationRange) % numMachines;
//             } 
//             else if (mutationStrategy == "decrement") {
//                 chromosome[index].machine = (chromosome[index].machine - mutationRange + numMachines) % numMachines;
//             }
//         }
//     }
// }

void mutation(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange, const vector<int>& tasks, double pressure) {
    random_device rd;
    mt19937 gen(rd());
    
    // 1. Calculate current machine loads and identify critical tasks
    vector<int> machineLoads(numMachines, 0);
    for (const auto& gene : chromosome) {
        machineLoads[gene.machine] += tasks[gene.task];
    }
    int currentCmax = *max_element(machineLoads.begin(), machineLoads.end());
    
    // 2. Determine critical machines (within 10% of Cmax)
    vector<int> criticalMachines;
    for (int m = 0; m < numMachines; ++m) {
        if (machineLoads[m] >= 0.9 * currentCmax) {
            criticalMachines.push_back(m);
        }
    }
    
    // 3. Mutate with probability based on machine criticality
    uniform_real_distribution<> probDist(0.0, 1.0);
    for (auto& gene : chromosome) {
        // Base mutation probability scaled by machine load
        double criticality = (double)machineLoads[gene.machine]/currentCmax;
        double mutationProb = mutationRate * (1.0 + pressure * criticality);
        
        if (probDist(gen) < mutationProb) {
            // Find best alternative machine
            int bestMachine = -1;
            int minLoad = INT_MAX;
            
            for (int m = 0; m < numMachines; ++m) {
                if (m != gene.machine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }
            
            if (bestMachine != -1) {
                // Update machine loads
                machineLoads[gene.machine] -= tasks[gene.task];
                gene.machine = bestMachine;
                machineLoads[bestMachine] += tasks[gene.task];
            }
        }
    }
}

vector<Gene> greedy(int numMachines, vector<int>& tasks) {
    vector<int> taskOrder(tasks.size());
    vector<int> machinesLoad(numMachines, 0); 
    vector<Gene> chromosome;

    iota(taskOrder.begin(), taskOrder.end(), 0);

    sort(taskOrder.begin(), taskOrder.end(),
        [&](int a, int b) {
            return tasks[a] > tasks[b];
        }
    );

    for (int task : taskOrder) {
        int minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back(Gene{task, minMachine});
        machinesLoad[minMachine] += tasks[task];
    }

    return chromosome;
}

int fitnessCalculation(int machines, const vector<Gene>& chromosome, const vector<int>& tasks) {
    vector<int> timesList(machines, 0);

    for (const auto& gene : chromosome) {
        if (gene.machine < 0 || gene.machine >= machines) {
            cerr << "Invalid machine number: " << gene.machine << endl;
            exit(1);
        }
        if (gene.task < 0 || gene.task >= tasks.size()) {
            cerr << "Invalid task number: " << gene.task << endl;
            exit(1);
        }
        timesList[gene.machine] += tasks[gene.task];
    }

    // Verification output
    // cout << "Machine loads: ";
    // for (int load : timesList) {
    //     cout << load << " ";
    // }
    // cout << endl;

    return *max_element(timesList.begin(), timesList.end());
}

pair<int, vector<int>> parseData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int numMachines, task_count;
    file >> numMachines >> task_count;

    vector<int> tasks(task_count);
    for (int i = 0; i < task_count; ++i) {
        if (!(file >> tasks[i])) {
            cerr << "Error reading task data" << endl;
            exit(1);
        }
    }
    return {numMachines, tasks};
}

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion, const vector<int>& tasks) {
    // Validate input chromosomes
    if (chromosome1.size() != chromosome2.size() || chromosome1.size() != tasks.size()) {
        cerr << "Error: Chromosome size mismatch in crossover" << endl;
        exit(1);
    }

    // Ensure proportion is within valid range
    proportion = max(0.1, min(0.9, proportion));
    int splitPoint = static_cast<int>(chromosome1.size() * proportion);
    
    // Ensure we don't split at the very beginning or end
    splitPoint = max(1, min(static_cast<int>(chromosome1.size()) - 1, splitPoint));

    // Create offspring
    vector<Gene> child1(chromosome1.begin(), chromosome1.begin() + splitPoint);
    vector<Gene> child2(chromosome2.begin(), chromosome2.begin() + splitPoint);
    
    // Second part: copy remaining tasks, ensuring no duplicates
    unordered_set<int> child1Tasks;
    unordered_set<int> child2Tasks;
    
    for (const auto& gene : child1) {
        child1Tasks.insert(gene.task);
    }
    for (const auto& gene : child2) {
        child2Tasks.insert(gene.task);
    }

    for (const auto& gene : chromosome2) {
        if (child1Tasks.find(gene.task) == child1Tasks.end()) {
            child1.push_back(gene);
            child1Tasks.insert(gene.task);
        }
    }

    for (const auto& gene : chromosome1) {
        if (child2Tasks.find(gene.task) == child2Tasks.end()) {
            child2.push_back(gene);
            child2Tasks.insert(gene.task);
        }
    }

    // Verify we have all tasks
    if (child1.size() != tasks.size() || child2.size() != tasks.size()) {
        cerr << "Error: Crossover produced invalid chromosome size" << endl;
        cerr << "Child1 size: " << child1.size() << " Child2 size: " << child2.size() << endl;
        exit(1);
    }

    return {child1, child2};
}

pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness) {
    vector<pair<vector<Gene>, int>> zipped;
    for (size_t i = 0; i < chromosomes.size(); ++i) {
        zipped.emplace_back(chromosomes[i], fitness[i]);
    }

    sort(zipped.begin(), zipped.end(), 
        [](const pair<vector<Gene>, int>& a, const pair<vector<Gene>, int>& b) {
            return a.second < b.second;
        }
    );

    for (size_t i = 0; i < zipped.size(); ++i) {
        chromosomes[i] = zipped[i].first;
        fitness[i] = zipped[i].second;
    }

    return {chromosomes, fitness};
}

pair<vector<vector<Gene>>, vector<int>> initialGeneration(vector<int> tasks, int chromosomesAmount, int numMachines) {
    vector<vector<Gene>> chromosomes;
    vector<int> fitness;

    // Add greedy solution first
    chromosomes.push_back(greedy(numMachines, tasks));
    fitness.push_back(fitnessCalculation(numMachines, chromosomes.back(), tasks));
    cout << "Greedy Cmax: " << fitness.back() << endl;

    // Generate random chromosomes
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> machineDist(0, numMachines - 1);

    for (int i = 1; i < chromosomesAmount; ++i) {
        vector<Gene> chromosome;
        for (int j = 0; j < tasks.size(); ++j) {
            chromosome.emplace_back(j, machineDist(gen));
        }
        chromosomes.push_back(chromosome);
        fitness.push_back(fitnessCalculation(numMachines, chromosome, tasks));
    }

    return sortChromosomes(chromosomes, fitness);
}

pair<vector<vector<Gene>>, vector<int>> evolution(vector<vector<Gene>>& chromosomes, vector<int>& fitness, 
                                                double mutationRate, int chromosomesPreserved, 
                                                int maxNewChromosomes, int numMachines, 
                                                vector<int>& tasks, double crossingProportion,
                                                double pressure) {
    random_device rd;
    mt19937 gen(rd());
    
    // Preserve best chromosomes
    vector<vector<Gene>> newPopulation(chromosomes.begin(), chromosomes.begin() + chromosomesPreserved);
    
    // Generate offspring through crossover
    uniform_int_distribution<> parentDist(0, chromosomes.size() - 1);
    int offspringNeeded = maxNewChromosomes;
    int offspringGenerated = 0;

    while (offspringGenerated < offspringNeeded) {
        int parent1 = parentDist(gen);
        int parent2 = parentDist(gen);
        
        auto [child1, child2] = crossing(chromosomes[parent1], chromosomes[parent2], crossingProportion, tasks);
        
        newPopulation.push_back(child1);
        offspringGenerated++;
        
        if (offspringGenerated < offspringNeeded) {
            newPopulation.push_back(child2);
            offspringGenerated++;
        }
    }

    // Apply mutation
    uniform_int_distribution<> mutationRangeDist(1, numMachines - 1);
    for (size_t i = chromosomesPreserved; i < newPopulation.size(); ++i) {
        mutation(mutationRate, newPopulation[i], numMachines, mutationRangeDist(gen), tasks, pressure);
    }

    // Evaluate fitness
    vector<int> newFitness;
    for (const auto& chromosome : newPopulation) {
        newFitness.push_back(fitnessCalculation(numMachines, chromosome, tasks));
    }

    return sortChromosomes(newPopulation, newFitness);
}

int main() {
    // Parameters
    double mutationRate = 0.35;
    int chromosomesAmount = 50;
    int chromosomesPreservedPercentage = 5;
    double crossingProportion = 0.5;
    int generations = 50000;
    double mutationPressure = 0.15;
    // Load data
    auto [numMachines, tasks] = parseData("../data/data.txt");
    if (numMachines <= 0 || tasks.empty()) {
        cerr << "Invalid input data" << endl;
        return 1;
    }

    
    int totalTaskTime = accumulate(tasks.begin(), tasks.end(), 0);
    int lowerBound = static_cast<int>(ceil(static_cast<double>(totalTaskTime) / numMachines));
    cout << "Approximate theoretical minimum Cmax: " << lowerBound << endl;

    int chromosomesPreserved = max(1, static_cast<int>(chromosomesAmount * chromosomesPreservedPercentage / 100.0));
    int maxNewChromosomes = chromosomesAmount - chromosomesPreserved;

    // Initial generation
    auto [chromosomes, fitness] = initialGeneration(tasks, chromosomesAmount, numMachines);
    BestChromosome bestChromosome = {chromosomes[0], fitness[0], 0};

    // Evolution loop
    for (int generation = 1; generation <= generations; ++generation) {
        if (generation % 1000 == 0) {
            cout << "GENERATION: " << generation << endl;
        }

        tie(chromosomes, fitness) = evolution(chromosomes, fitness, mutationRate, 
                                           chromosomesPreserved, maxNewChromosomes,
                                           numMachines, tasks, crossingProportion,
                                           mutationPressure);

        // Track best solution
        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome = {chromosomes[0], fitness[0], generation};
            cout << "Generation " << generation << ": New best Cmax = " << bestChromosome.fitness << endl;
        }

        // Early stopping if optimal found
        if (bestChromosome.fitness == 6) {
            cout << "Optimal solution found!" << endl;
            break;
        }
        


    }

    // Output final results
    cout << "\nFinal Results:" << endl;
    cout << "Best Cmax: " << bestChromosome.fitness << endl;
    cout << "Found in generation: " << bestChromosome.generation << endl;
    
    // Print machine loads for verification
    vector<int> machineLoads(numMachines, 0);
    for (const auto& gene : bestChromosome.chromosome) {
        machineLoads[gene.machine] += tasks[gene.task];
    }
    // cout << "Machine loads: ";
    // for (int load : machineLoads) {
    //     cout << load << " ";
    // }
    // cout << endl;


    return 0;
}
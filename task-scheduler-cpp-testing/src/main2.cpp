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

using namespace std;

struct Gene {
    int task;
    int machine;
};

struct BestChromosome {
    vector<Gene> chromosome;
    int fitness;
    int generation;
};

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion);
pair<int, vector<int>> parseData(const string& filename);
int fitnessCalculation(int numMachines, vector<Gene>& chromosome, vector<int>& tasks);
vector<Gene> greedy(int numMachines, vector<int>& tasks);
void mutation_inplace(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange);

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

vector<Gene> greedy(int numMachines, vector<int>& tasks) {
    vector<int> taskOrder(tasks.size());
    vector<int> machinesLoad(numMachines, 0); 
    vector<Gene> chromosome;

    int minMachine = -1;

    iota(taskOrder.begin(), taskOrder.end(), 0);

    sort(taskOrder.begin(), taskOrder.end(),
        [&](int a, int b) {
            return tasks[a] > tasks[b];
        }
    );

    for (int task : taskOrder) {
        minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back({task, minMachine + 1});
        machinesLoad[minMachine] += tasks[task]; 
    }

    return chromosome;
}

int fitnessCalculation(int machines, vector<Gene>& chromosome, vector<int>& tasks) {
    vector<int> timesList(machines, 0);

    for (int i = 0; i < chromosome.size(); i++) {
        int index = chromosome[i].machine - 1;
        timesList[index] += chromosome[i].task;
    }

    int Cmax = *max_element(timesList.begin(), timesList.end());

    return Cmax;
}

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

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion) {
    proportion = max(0.1, min(0.9, proportion));
    int splitPoint = static_cast<int>(chromosome1.size() * proportion);
    vector<Gene> newChromosome1 = vector<Gene>(chromosome1.begin(), chromosome1.begin() + splitPoint);
    newChromosome1.insert(newChromosome1.end(), chromosome2.begin() + splitPoint, chromosome2.end());

    vector<Gene> newChromosome2 = vector<Gene>(chromosome2.begin(), chromosome2.begin() + splitPoint);
    newChromosome2.insert(newChromosome2.end(), chromosome1.begin() + splitPoint, chromosome1.end());
    return {newChromosome1, newChromosome2};
}


pair<vector<vector<Gene>>, vector<int>> sortChromosomes(vector<vector<Gene>> chromosomes, vector<int> fitness) {
    vector<pair<vector<Gene>, int>> zipped;
    for (size_t i = 0; i < chromosomes.size(); ++i) {
        zipped.push_back({chromosomes[i], fitness[i]});
    }

    sort(zipped.begin(), zipped.end(), 
        [](const pair<vector<Gene>, int> a, const pair<vector<Gene>, int> b) {
            return a.second < b.second;
        }
    );

    for (size_t i = 0; i < zipped.size(); ++i) {
        chromosomes[i] = zipped[i].first;
        fitness[i] = zipped[i].second;
    }

    return {chromosomes, fitness};
}

pair<vector<vector<Gene>>, vector<int>> initialGeneration(vector<int> tasks, int chromosomesAmount, vector<vector<Gene>> chromosomes, int numMachines) {
    vector<int> fitness;
    chromosomes.push_back(greedy(numMachines, tasks));
    random_device rd;
    mt19937 gen(rd());
    int Cmax = fitnessCalculation(numMachines, chromosomes[0], tasks);
    cout << "Greedy: " << Cmax << endl;
    int tempMax = 999999999;
    for (int i = 0; i < chromosomesAmount - 1; ++i) {
        vector<Gene> chromosome;
        for (int task : tasks) {
            uniform_int_distribution machineDist(1, numMachines);
            chromosome.push_back({task, machineDist(gen)});
        }
        chromosomes.push_back(chromosome);
    }

    for (int i = 1; i < chromosomesAmount; ++i) {
        fitness.push_back(fitnessCalculation(numMachines, chromosomes[i], tasks));   
    }

    return {chromosomes, fitness};
}

pair<vector<vector<Gene>>, vector<int>> evolution(vector<vector<Gene>>& chromosomes, vector<int>& fitness, double mutationRate, int chromosomesPreserved, int maxNewChromosomes, int numMachines, vector<int>& tasks, int chromosomesAmount, double crossingProportion) {
    random_device rd;
    mt19937 gen(rd());
    
    vector<vector<Gene>> bestChromosomes(chromosomes.begin(), chromosomes.begin() + chromosomesPreserved);
    vector<vector<Gene>> newChromosomes;
    
    uniform_int_distribution<> indexDist(0, chromosomesAmount - 1);

    // int pairs = maxNewChromosomes / 2 + (maxNewChromosomes % 2);
    int pairs = (maxNewChromosomes + 1) / 2;

    for (int i = 0; i < pairs; ++i) {
        int index1 = indexDist(gen);
        int index2 = indexDist(gen);
        vector<Gene> child1;
        vector<Gene> child2;
        tie(child1, child2) = crossing(chromosomes[index1], chromosomes[index2], crossingProportion);

        newChromosomes.push_back(child1);
        if (newChromosomes.size() < maxNewChromosomes) {
            newChromosomes.push_back(child2);
        }
    }
    uniform_int_distribution<> mutationRangeDist(1, numMachines);

    for (int i = 0; i < newChromosomes.size(); ++i) {
        mutation_inplace(mutationRate, newChromosomes[i], numMachines, mutationRangeDist(gen));
    }
    
    chromosomes = bestChromosomes;
    chromosomes.insert(chromosomes.end(), newChromosomes.begin(), newChromosomes.end());

    fitness.clear();

    for (vector<Gene> chromosome : chromosomes) {
        fitness.push_back(fitnessCalculation(numMachines, chromosome, tasks));
    }

    return sortChromosomes(chromosomes, fitness);
}

int main() {
    double mutationRate = 0.25;
    int chromosomesAmount = 20;
    int chromosomesPreservedPrecentage = 5;
    double crossingProportion = 0.25;
    int generations = 50000;
    int maxTime = 10000; // In miliseconds

    int chromosomesPreserved = ceil(chromosomesAmount * (chromosomesPreservedPrecentage / 100));
    int maxNewChromosomes = chromosomesAmount - chromosomesPreserved;
    vector<vector<Gene>> chromosomes;
    vector<int> fitness;
    int numMachines;
    vector<int> tasks;

    tie(numMachines, tasks) = parseData("../data/data.txt");
    if (numMachines <= 0) {
        std::cerr << "Error: Invalid number of machines (must be greater than 0)." << std::endl;
        exit(1); 
    }
    
    tie(chromosomes, fitness) = initialGeneration(tasks, chromosomesAmount, chromosomes, numMachines);
    sortChromosomes(chromosomes, fitness);

    BestChromosome bestChromosome = {chromosomes[0], fitnessCalculation(numMachines, chromosomes[0], tasks), 1};
    for (int i = 1; i <= generations; ++i) {
        tie(chromosomes, fitness) = evolution(chromosomes, fitness, mutationRate, chromosomesPreserved, maxNewChromosomes, numMachines, tasks, chromosomesAmount, crossingProportion);
        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome.chromosome = chromosomes[0];
            bestChromosome.fitness = fitness[0];
            bestChromosome.generation = i;
            cout << "Current Cmax: " << bestChromosome.fitness << " found in generation: " << bestChromosome.generation << endl << flush;
        }
    }

    cout << "Cmax" << bestChromosome.fitness << endl;
    return 0;
}
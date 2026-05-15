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
#include <climits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <cstdlib>

#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct Config {
    float mutationProbability;
    int populationsSize;
    float chromosomesPreservedPercentage;
    float crossoverRatio;
    int generations;
    float mutationPressure;
    string dataFile;
    int maxTime;
};

__device__ __constant__ int numMachines;
__device__ __constant__ int numTasks;
__device__ __constant__ int taskDurations[2048]; 

struct Gene {
    int task;
    int machine;
};

struct Chromosome {
    Gene genes[2048]; 
    int fitness;
};

extern "C" __global__ void geneticAlgorithm(Chromosome* population, Chromosome* bestChromosome, int* generation, int* stopFlag, unsigned int seed, Config config);
__device__ void initializePopulation(Chromosome* population, unsigned int* seed, Config config);
__device__ void calculateFitness(Chromosome* chromosome);
__device__ void performEvolution(Chromosome* population, unsigned int* seed, Config config);
__device__ void crossover(const Chromosome* parent1, const Chromosome* parent2, Chromosome* child1, Chromosome* child2, unsigned int* seed, float crossoverRatio);
__device__ void mutate(Chromosome* chromosome, unsigned int* seed, float mutationProbability, float mutationPressure);
__device__ void greedyInitialize(Chromosome* chromosome);

__global__ void initializePopulationKernel(Chromosome* population, unsigned int seed, Config config);
__global__ void evaluatePopulationKernel(Chromosome* population, int populationSize);
__global__ void evolvePopulationKernel(Chromosome* population, Chromosome* bestChromosome, int* generation, unsigned int seed, Config config);


__device__ unsigned int randInt(unsigned int* seed);
Config loadConfig(const string& configFile);
void loadData(const string& filename, int* h_numMachines, int* h_numTasks, int* h_taskDurations);

int main() {
    Config config = loadConfig("config.json");
    int h_numMachines;
    int h_numTasks;
    int h_taskDurations[2048];
    loadData(config.dataFile, &h_numMachines, &h_numTasks, h_taskDurations);
    cudaError_t err;
    err = cudaMemcpyToSymbol(numMachines, &h_numMachines, sizeof(int));
    if (err != cudaSuccess) { printf("MemcpyToSymbol error: %s\n", cudaGetErrorString(err)); }
    err = cudaMemcpyToSymbol(numTasks, &h_numTasks, sizeof(int));
    if (err != cudaSuccess) { printf("MemcpyToSymbol error: %s\n", cudaGetErrorString(err)); }
    err = cudaMemcpyToSymbol(taskDurations, h_taskDurations, sizeof(int) * h_numTasks);
    if (err != cudaSuccess) { printf("MemcpyToSymbol error: %s\n", cudaGetErrorString(err)); }

    printf("numTasks: %d, taskDurations[0]: %d\n", h_numTasks, h_taskDurations[0]);

    Chromosome* d_population;
    Chromosome* d_bestChromosome;
    int* d_generation;
    int* d_stopFlag;
    cudaMalloc(&d_population, sizeof(Chromosome) * config.populationsSize);
    cudaMalloc(&d_bestChromosome, sizeof(Chromosome));
    cudaMalloc(&d_generation, sizeof(int));
    cudaMalloc(&d_stopFlag, sizeof(int));
    int h_stopFlag = 0;
    cudaMemcpy(d_stopFlag, &h_stopFlag, sizeof(int), cudaMemcpyHostToDevice);
    int h_generation = 0;
    cudaMemcpy(d_generation, &h_generation, sizeof(int), cudaMemcpyHostToDevice);
    Config* d_config;
    cudaMalloc(&d_config, sizeof(Config));
    cudaMemcpy(d_config, &config, sizeof(Config), cudaMemcpyHostToDevice);
    unsigned int seed = time(nullptr);
    auto startTime = chrono::steady_clock::now();
    unsigned int seed = time(nullptr);

    int blockSize = 256;
    int gridSize = (config.populationsSize + blockSize - 1) / blockSize;
    geneticAlgorithm<<<gridSize, blockSize>>>(d_population, d_bestChromosome, d_generation, d_stopFlag, seed, config);
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(cudaErr));
    }
    cudaDeviceSynchronize();
    Chromosome h_bestChromosome;
    cudaMemcpy(&h_bestChromosome, d_bestChromosome, sizeof(Chromosome), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_generation, d_generation, sizeof(int), cudaMemcpyDeviceToHost);
    auto endTime = chrono::steady_clock::now();
    chrono::duration<double> elapsed = endTime - startTime;
    cout << "\nFinal Results:" << endl;
    cout << "Best Cmax: " << h_bestChromosome.fitness << endl;
    cout << "Found in generation: " << h_generation << endl;
    int totalTaskTime = 0;
    for (int i = 0; i < h_numTasks; i++) {
        totalTaskTime += h_taskDurations[i];
    }
    int lowerBound = static_cast<int>(ceil(static_cast<double>(totalTaskTime) / h_numMachines));
    cout << "Lower bound: " << lowerBound << endl;
    cout << "Elapsed time: " << elapsed.count() << "s" << endl;
    cudaFree(d_population);
    cudaFree(d_bestChromosome);
    cudaFree(d_generation);
    cudaFree(d_stopFlag);
    cudaFree(d_config);
    return 0;
}

Config loadConfig(const string& configFile) {
    ifstream input(configFile);
    if (!input.is_open()) {
        cerr << "Could not open config file: " << configFile << endl;
        exit(1);
    }
    json j;
    input >> j;
    Config config;
    config.mutationProbability = j.value("mutationProbability", 0.35f);
    config.populationsSize = j.value("populationsSize", 50);
    config.chromosomesPreservedPercentage = j.value("chromosomesPreservedPercentage", 5.0f) / 100.0f;
    config.crossoverRatio = j.value("crossoverRatio", 0.5f);
    config.generations = j.value("generations", 50000);
    config.mutationPressure = j.value("mutationPressure", 0.15f);
    config.dataFile = j.value("dataFile", "data/data.txt");
    config.maxTime = j.value("maxTime", 300);
    return config;
}

__global__ void geneticAlgorithm(Chromosome* population, Chromosome* bestChromosome, int* generation, int* stopFlag, unsigned int seed, Config config) {
    initializePopulation(population, &seed, config);
    for (int i = 0; i < config.populationsSize; i++) {
        calculateFitness(&population[i]);
    }
    for (int i = 0; i < config.populationsSize - 1; i++) {
        for (int j = 0; j < config.populationsSize - i - 1; j++) {
            if (population[j].fitness > population[j + 1].fitness) {
                Chromosome temp = population[j];
                population[j] = population[j + 1];
                population[j + 1] = temp;
            }
        }
    }
    *bestChromosome = population[0];
    int chromosomesPreserved = max(1, (int)(config.populationsSize * config.chromosomesPreservedPercentage));
    
    while (*generation < config.generations && !(*stopFlag)) {
        if (*generation == 0) {
            printf("Initial best fitness: %d\n", population[0].fitness);
        }
        (*generation)++;
        performEvolution(population, &seed, config);
        if (population[0].fitness < bestChromosome->fitness) {
            *bestChromosome = population[0];
        }
        if (*generation % 1000 == 0) { }
    }
}

__device__ void initializePopulation(Chromosome* population, unsigned int* seed, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.populationsSize) return;
    
    if (idx == 0) {
        greedyInitialize(&population[0]);
    } else {
        for (int j = 0; j < numTasks; j++) {
            population[idx].genes[j].task = j;
            population[idx].genes[j].machine = randInt(seed) % numMachines;
        }
    }
    calculateFitness(&population[idx]);
}

__device__ void calculateFitness(Chromosome* chromosome) {
    int machineLoads[50] = {0};
    for (int i = 0; i < numTasks; i++) {
        int machine = chromosome->genes[i].machine;
        int taskId = chromosome->genes[i].task;
        machineLoads[machine] += taskDurations[taskId];
    }
    int maxLoad = 0;
    for (int i = 0; i < numMachines; i++) {
        if (machineLoads[i] > maxLoad) {
            maxLoad = machineLoads[i];
        }
    }
    chromosome->fitness = maxLoad;
}

__device__ void performEvolution(Chromosome* population, unsigned int* seed, Config config) {
    Chromosome newPopulation[2048];
    int chromosomesPreserved = max(1, (int)(config.populationsSize * config.chromosomesPreservedPercentage));
    for (int i = 0; i < chromosomesPreserved; i++) {
        newPopulation[i] = population[i];
    }
    int offspringCount = chromosomesPreserved;
    while (offspringCount < config.populationsSize) {
        int parent1 = randInt(seed) % config.populationsSize;
        int parent2 = randInt(seed) % config.populationsSize;
        if (parent1 == parent2) continue;
        crossover(&population[parent1], &population[parent2], &newPopulation[offspringCount], (offspringCount + 1 < config.populationsSize) ? &newPopulation[offspringCount + 1] : nullptr, seed, config.crossoverRatio);
        offspringCount += 2;
    }
    for (int i = chromosomesPreserved; i < config.populationsSize; i++) {
        if ((randInt(seed) % 10000) / 10000.0f < config.mutationProbability) {
            mutate(&newPopulation[i], seed, config.mutationProbability, config.mutationPressure);
        }
        calculateFitness(&newPopulation[i]);
    }
    for (int i = 0; i < config.populationsSize - 1; i++) {
        for (int j = 0; j < config.populationsSize - i - 1; j++) {
            if (newPopulation[j].fitness > newPopulation[j + 1].fitness) {
                Chromosome temp = newPopulation[j];
                newPopulation[j] = newPopulation[j + 1];
                newPopulation[j + 1] = temp;
            }
        }
    }
    for (int i = 0; i < config.populationsSize; i++) {
        population[i] = newPopulation[i];
    }
}

__device__ void crossover(const Chromosome* parent1, const Chromosome* parent2, Chromosome* child1, Chromosome* child2, unsigned int* seed, float crossoverRatio) {
    float proportion = crossoverRatio;
    int splitPoint = (int)(numTasks * proportion);
    bool child1Tasks[2048] = {false};
    bool child2Tasks[2048] = {false};
    for (int i = 0; i < splitPoint; i++) {
        child1->genes[i] = parent1->genes[i];
        child1Tasks[child1->genes[i].task] = true;
        if (child2) {
            child2->genes[i] = parent2->genes[i];
            child2Tasks[child2->genes[i].task] = true;
        }
    }
    int child1Pos = splitPoint;
    int child2Pos = splitPoint;
    for (int i = 0; i < numTasks; i++) {
        if (!child1Tasks[parent2->genes[i].task]) {
            child1->genes[child1Pos++] = parent2->genes[i];
            child1Tasks[parent2->genes[i].task] = true;
        }
        if (child2 && !child2Tasks[parent1->genes[i].task]) {
            child2->genes[child2Pos++] = parent1->genes[i];
            child2Tasks[parent1->genes[i].task] = true;
        }
    }
}

__device__ void mutate(Chromosome* chromosome, unsigned int* seed, float mutationProbability, float mutationPressure) {
    int machineLoads[50] = {0}; 
    for (int i = 0; i < numTasks; i++) {
        machineLoads[chromosome->genes[i].machine] += taskDurations[chromosome->genes[i].task];
    }
    int currentCmax = 0;
    for (int i = 0; i < numMachines; i++) {
        if (machineLoads[i] > currentCmax) {
            currentCmax = machineLoads[i];
        }
    }
    for (int i = 0; i < numTasks; i++) {
        int machine = chromosome->genes[i].machine;
        double criticality = (double)machineLoads[machine] / currentCmax;
        double mutationProb = mutationProbability * (1.0 + mutationPressure * criticality);
        if ((randInt(seed) % 10000) / 10000.0f < mutationProb) {
            int oldMachine = machine;
            int oldLoad = machineLoads[oldMachine];
            int bestMachine = -1;
            int minLoad = INT_MAX;
            for (int m = 0; m < numMachines; m++) {
                if (m != oldMachine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }
            if (bestMachine != -1) {
                int newLoad = machineLoads[bestMachine];
                int updatedOldLoad = oldLoad - taskDurations[chromosome->genes[i].task];
                int updatedNewLoad = newLoad + taskDurations[chromosome->genes[i].task];
                int localOldMax = max(machineLoads[oldMachine], machineLoads[bestMachine]);
                int localNewMax = max(updatedOldLoad, updatedNewLoad);
                if (localNewMax <= localOldMax) {
                    machineLoads[oldMachine] = updatedOldLoad;
                    machineLoads[bestMachine] = updatedNewLoad;
                    chromosome->genes[i].machine = bestMachine;
                }
            }
        }
    }
}

__device__ void greedyInitialize(Chromosome* chromosome) {
    int machinesLoad[50] = {0}; 
    int taskOrder[2048];
    for (int i = 0; i < numTasks; i++) {
        taskOrder[i] = i;
    }
    for (int i = 0; i < numTasks - 1; i++) {
        for (int j = 0; j < numTasks - i - 1; j++) {
            if (taskDurations[taskOrder[j]] < taskDurations[taskOrder[j + 1]]) {
                int temp = taskOrder[j];
                taskOrder[j] = taskOrder[j + 1];
                taskOrder[j + 1] = temp;
            }
        }
    }
    for (int i = 0; i < numTasks; i++) {
        int task = taskOrder[i];
        int minMachine = 0;
        for (int m = 1; m < numMachines; m++) {
            if (machinesLoad[m] < machinesLoad[minMachine]) {
                minMachine = m;
            }
        }
        chromosome->genes[i].task = task;
        chromosome->genes[i].machine = minMachine;
        machinesLoad[minMachine] += taskDurations[task];
    }
}

__device__ unsigned int randInt(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

void loadData(const string& filename, int* h_numMachines, int* h_numTasks, int* h_taskDurations) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    file >> *h_numMachines >> *h_numTasks;
    for (int i = 0; i < *h_numTasks; ++i) {
        if (!(file >> h_taskDurations[i])) {
            cerr << "Error reading task data" << endl;
            exit(1);
        }
    }
    cout << "Loaded data: " << *h_numMachines << " machines, " << *h_numTasks << " tasks" << endl;
}
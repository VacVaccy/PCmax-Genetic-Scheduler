#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <climits>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

#define MAX_MACHINES_PER_CHROMOSOME 256
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct Gene {
    int task;
    int machine;
    Gene(int t, int m) : task(t), machine(m) {}
};

struct BestChromosome {
    vector<Gene> chromosome;
    int fitness;
    int generation;
    BestChromosome() : fitness(INT_MAX), generation(0) {}
};

struct Config {
    double mutationProbability;
    int populationsSize;
    int chromosomesPreservedPercentage;
    double crossoverRatio;
    int generations;
    double mutationPressure;
    string dataFile;
    int maxTime;
};

Config loadConfig(const string& configFile) {
    ifstream input(configFile);
    if (!input.is_open()) {
        cerr << "Could not open config file: " << configFile << endl;
        exit(1);
    }
    json j;
    input >> j;
    Config config;
    config.mutationProbability = j.value("mutationProbability", 0.35);
    config.populationsSize = j.value("populationsSize", 50);
    config.chromosomesPreservedPercentage = j.value("chromosomesPreservedPercentage", 5);
    config.crossoverRatio = j.value("crossoverRatio", 0.5);
    config.generations = j.value("generations", 50000);
    config.mutationPressure = j.value("mutationPressure", 0.15);
    config.dataFile = j.value("dataFile", "../data/data.txt");
    config.maxTime = j.value("maxTime", 300);
    return config;
}

pair<int, vector<int>> parseData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    int numMachines, task_count;
    file >> numMachines >> task_count;
    vector<int> taskDurations(task_count);
    for (int i = 0; i < task_count; ++i) {
        file >> taskDurations[i];
    }
    return make_pair(numMachines, taskDurations);
}

vector<Gene> greedySchedule(int numMachines, const vector<int>& taskDurations) {
    vector<int> taskOrder(taskDurations.size());
    vector<int> machinesLoad(numMachines, 0);
    vector<Gene> chromosome;
    iota(taskOrder.begin(), taskOrder.end(), 0);
    sort(taskOrder.begin(), taskOrder.end(), [&](int a, int b) {
        return taskDurations[a] > taskDurations[b];
    });
    for (int task : taskOrder) {
        int minMachine = distance(machinesLoad.begin(), min_element(machinesLoad.begin(), machinesLoad.end()));
        chromosome.push_back(Gene(task, minMachine));
        machinesLoad[minMachine] += taskDurations[task];
    }
    return chromosome;
}

int calculateFitness(int numMachines, const vector<Gene>& chromosome, const vector<int>& taskDurations) {
    vector<int> machineTimes(numMachines, 0);
    for (const auto& gene : chromosome) {
        machineTimes[gene.machine] += taskDurations[gene.task];
    }
    return *max_element(machineTimes.begin(), machineTimes.end());
}

void sortPopulation(vector<vector<Gene>>& population, vector<int>& fitness) {
    vector<pair<int, size_t>> fitnessIndices;
    for (size_t i = 0; i < fitness.size(); ++i) {
        fitnessIndices.push_back(make_pair(fitness[i], i));
    }
    sort(fitnessIndices.begin(), fitnessIndices.end());
    vector<vector<Gene>> sortedPopulation;
    vector<int> sortedFitness;
    for (const auto& fi : fitnessIndices) {
        sortedPopulation.push_back(population[fi.second]);
        sortedFitness.push_back(fi.first);
    }
    population = sortedPopulation;
    fitness = sortedFitness;
}

pair<vector<vector<Gene>>, vector<int>> createInitialPopulation(const vector<int>& taskDurations, int populationSize, int numMachines) {
    vector<vector<Gene>> population;
    vector<int> fitness;
    auto greedySolution = greedySchedule(numMachines, taskDurations);
    population.push_back(greedySolution);
    fitness.push_back(calculateFitness(numMachines, greedySolution, taskDurations));
    cout << "Greedy Cmax: " << fitness.back() << endl;
    mt19937 gen(random_device{}());
    uniform_int_distribution<int> machineDist(0, numMachines - 1);
    for (int i = 1; i < populationSize; ++i) {
        vector<Gene> chromosome;
        for (int j = 0; j < taskDurations.size(); ++j) {
            chromosome.push_back(Gene(j, machineDist(gen)));
        }
        population.push_back(chromosome);
        fitness.push_back(calculateFitness(numMachines, chromosome, taskDurations));
    }
    sortPopulation(population, fitness);
    return make_pair(population, fitness);
}

__global__ void calculateFitnessKernel(int* chromosomes, const int* taskDurations, int* fitness, int numMachines, int numTasks, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= populationSize) return;
    int machineLoads[MAX_MACHINES_PER_CHROMOSOME] = {0};
    for (int i = 0; i < numTasks; ++i) {
        int machine = chromosomes[idx * numTasks + i];
        machineLoads[machine] += taskDurations[i];
    }
    int maxLoad = 0;
    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > maxLoad) maxLoad = machineLoads[i];
    }
    fitness[idx] = maxLoad;
}

__global__ void mutationKernel(int* chromosomes, const int* taskDurations, int numMachines, 
                             int numTasks, int populationSize, double mutationProbBase, 
                             double mutationPressure, curandState* states) {
    int individualIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (individualIdx >= populationSize) return;

    curandState localState = states[individualIdx];
    
    // Oblicz obciążenie maszyn dla tego osobnika
    int machineLoads[MAX_MACHINES_PER_CHROMOSOME] = {0};
    for (int i = 0; i < numTasks; ++i) {
        int machine = chromosomes[individualIdx * numTasks + i];
        machineLoads[machine] += taskDurations[i];
    }

    // Znajdź maksymalne obciążenie (Cmax)
    int currentCmax = 0;
    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > currentCmax) currentCmax = machineLoads[i];
    }

    // Dla każdego zadania w chromosomie
    for (int taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
        int oldMachine = chromosomes[individualIdx * numTasks + taskIdx];
        int taskDuration = taskDurations[taskIdx];
        
        // Oblicz prawdopodobieństwo mutacji
        double criticality = (double)machineLoads[oldMachine] / currentCmax;
        double mutationProb = mutationProbBase * (1.0 + mutationPressure * criticality);
        
        if (curand_uniform_double(&localState) < mutationProb) {
            // Znajdź najlepszą maszynę do przeniesienia
            int bestMachine = -1;
            int minLoad = INT_MAX;
            
            for (int m = 0; m < numMachines; ++m) {
                if (m != oldMachine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }

            // Jeśli znaleziono lepszą maszynę
            if (bestMachine != -1) {
                // ZMIANA: Usuń warunek sprawdzający poprawę
                machineLoads[oldMachine] -= taskDuration;
                machineLoads[bestMachine] += taskDuration;
                chromosomes[individualIdx * numTasks + taskIdx] = bestMachine;
                
                // OPTYMALIZACJA: Aktualizuj currentCmax
                if (machineLoads[bestMachine] > currentCmax) {
                    currentCmax = machineLoads[bestMachine];
                }
            }
        }
    }
    
    // Zapisz stan generatora liczb losowych
    states[individualIdx] = localState;
}

__device__ void orderBasedCrossover(const int* parent1, const int* parent2, int* child, int numTasks, curandState* state, double crossoverRatio) {
    int splitPoint = ceil(numTasks * (crossoverRatio));
    
    for (int i = 0; i < splitPoint; ++i) {
        child[i] = parent1[i];
    }

    bool used[MAX_MACHINES_PER_CHROMOSOME] = {false};
    for (int i = 0; i < splitPoint; ++i) {
        used[parent1[i]] = true;
    }

    int childPos = splitPoint;
    for (int i = 0; i < numTasks && childPos < numTasks; ++i) {
        if (!used[parent2[i]]) {
            child[childPos++] = parent2[i];
            used[parent2[i]] = true;
        }
    }
}

__global__ void crossoverKernel(const int* parents, int* offspring, int numTasks, int populationSize, curandState* states, double crossoverRatio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    curandState localState = states[idx];
    int parent1 = curand(&localState) % populationSize;
    int parent2 = curand(&localState) % (populationSize - 1);
    if (parent2 >= parent1) parent2++;

    orderBasedCrossover(
        &parents[parent1 * numTasks],
        &parents[parent2 * numTasks],
        &offspring[idx * numTasks],
        numTasks,
        &localState,
        crossoverRatio
    );
}

__global__ void initCurandStates(unsigned int seed, curandState* states, int numStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates) curand_init(seed, idx, 0, &states[idx]);
}

void runGeneticAlgorithmOnGPU(const Config& config, int numMachines, const vector<int>& taskDurations, BestChromosome& bestChromosome) {
    if (numMachines > MAX_MACHINES_PER_CHROMOSOME) {
        cerr << "Error: Too many machines (" << numMachines << " > " << MAX_MACHINES_PER_CHROMOSOME << ")\n";
        exit(EXIT_FAILURE);
    }
    
    int numTasks = taskDurations.size();
    int populationSize = config.populationsSize;
    int preservedCount = max(1, populationSize * config.chromosomesPreservedPercentage / 100);
    
    pair<vector<vector<Gene>>, vector<int>> initialPopulation = createInitialPopulation(taskDurations, populationSize, numMachines);
    vector<vector<Gene>>& population = initialPopulation.first;
    vector<int>& fitness = initialPopulation.second;
    
    if (fitness[0] < bestChromosome.fitness) {
        bestChromosome.chromosome = population[0];
        bestChromosome.fitness = fitness[0];
        bestChromosome.generation = 0;
    }
    
    // Alokacja pamięci na GPU
    int* d_population;
    int* d_taskDurations;
    int* d_fitness;
    int* d_offspring;
    curandState* d_randStates;
    
    CUDA_CHECK(cudaMalloc(&d_population, populationSize * numTasks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_taskDurations, numTasks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fitness, populationSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offspring, populationSize * numTasks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_randStates, populationSize * sizeof(curandState)));
    
    // Kopiujemy dane zadań tylko raz
    CUDA_CHECK(cudaMemcpy(d_taskDurations, taskDurations.data(), numTasks * sizeof(int), cudaMemcpyHostToDevice));
    
    // Inicjalizacja generatorów liczb losowych
    unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
    initCurandStates<<<(populationSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(seed, d_randStates, populationSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Przygotowanie płaskiej populacji początkowej
    vector<int> flatPopulation(populationSize * numTasks);
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < numTasks; ++j) {
            flatPopulation[i * numTasks + j] = population[i][j].machine;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_population, flatPopulation.data(), flatPopulation.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    auto startTime = chrono::steady_clock::now();
    
    for (int gen = 1; gen <= config.generations; ++gen) {
        auto elapsed = chrono::duration<double>(chrono::steady_clock::now() - startTime).count();
        if (elapsed >= config.maxTime) {
            cout << "\nTime limit reached after " << elapsed << "s\n";
            break;
        }
        
        if (gen % 1000 == 0 || gen == 1) {
            cout << "Generation " << gen << " | Best Cmax: " << bestChromosome.fitness 
                 << " | Time: " << fixed << setprecision(2) << elapsed << "s" << endl;
        }
        
        // Usunięto zbędne kopiowanie danych wejściowych w każdej iteracji
        
        crossoverKernel<<<(populationSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_population, d_offspring, numTasks, populationSize, d_randStates, config.crossoverRatio);
        
        mutationKernel<<<(populationSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_offspring, d_taskDurations, numMachines, numTasks, populationSize, 
            config.mutationProbability, config.mutationPressure, d_randStates);
        
        calculateFitnessKernel<<<(populationSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_offspring, d_taskDurations, d_fitness, numMachines, numTasks, populationSize);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Kopiujemy tylko wyniki fitness
        vector<int> offspringFitness(populationSize);
        CUDA_CHECK(cudaMemcpy(offspringFitness.data(), d_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Kopiujemy tylko offspring (potomków) z GPU
        vector<int> flatOffspring(populationSize * numTasks);
        CUDA_CHECK(cudaMemcpy(flatOffspring.data(), d_offspring, flatOffspring.size() * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Tworzenie nowej generacji
        vector<vector<Gene>> nextGen(population.begin(), population.begin() + preservedCount);
        vector<int> nextGenFitness(fitness.begin(), fitness.begin() + preservedCount);
        
        for (int i = 0; i < populationSize; ++i) {
            vector<Gene> chromosome;
            for (int j = 0; j < numTasks; ++j) {
                chromosome.push_back(Gene(j, flatOffspring[i * numTasks + j]));
            }
            nextGen.push_back(chromosome);
            nextGenFitness.push_back(offspringFitness[i]);
        }
        
        sortPopulation(nextGen, nextGenFitness);
        population.assign(nextGen.begin(), nextGen.begin() + populationSize);
        fitness.assign(nextGenFitness.begin(), nextGenFitness.begin() + populationSize);
        
        // Aktualizacja najlepszego chromosomu
        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome.chromosome = population[0];
            bestChromosome.fitness = fitness[0];
            bestChromosome.generation = gen;
            cout << "Generation " << gen << " | Best Cmax: " << bestChromosome.fitness << " | Time: " << fixed << setprecision(2) << elapsed << "s" << endl;
        }
        
        // Aktualizacja płaskiej populacji na GPU (kopiujemy tylko najlepsze osobniki)
        for (int i = 0; i < populationSize; ++i) {
            for (int j = 0; j < numTasks; ++j) {
                flatPopulation[i * numTasks + j] = population[i][j].machine;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_population, flatPopulation.data(), flatPopulation.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Zwolnienie pamięci GPU
    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_taskDurations));
    CUDA_CHECK(cudaFree(d_fitness));
    CUDA_CHECK(cudaFree(d_offspring));
    CUDA_CHECK(cudaFree(d_randStates));
}

int main() {
    Config config = loadConfig("config.json");
    pair<int, vector<int>> data = parseData(config.dataFile);
    int numMachines = data.first;
    vector<int> taskDurations = data.second;
    if (numMachines <= 0 || taskDurations.empty()) {
        cerr << "Invalid input data" << endl;
        return 1;
    }
    BestChromosome bestSolution;
    runGeneticAlgorithmOnGPU(config, numMachines, taskDurations, bestSolution);
    int totalTime = accumulate(taskDurations.begin(), taskDurations.end(), 0);
    int lowerBound = (totalTime + numMachines - 1) / numMachines;
    cout << "\n--- Results ---" << endl;
    cout << "Best Cmax: " << bestSolution.fitness << endl;
    cout << "Found in generation: " << bestSolution.generation << endl;
    cout << "Lower bound: " << lowerBound << endl;
    cout << "Optimality gap: " << 100.0 * (bestSolution.fitness - lowerBound) / lowerBound << "%" << endl;
    return 0;
}
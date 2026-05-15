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
#include <curand_kernel.h>
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

#define MAX_MACHINES 256
#define MAX_TASKS 4096

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

struct Gene {
    int task;
    int machine;

    __host__ __device__ Gene() : task(0), machine(0) {}
    __host__ __device__ Gene(int t, int m) : task(t), machine(m) {}
};

struct BestChromosome {
    vector<Gene> chromosome;
    int fitness;
    int generation;
    BestChromosome(vector<Gene> chrom, int fit, int gen) : chromosome(chrom), fitness(fit), generation(gen) {}
};

struct GPUData {
    Gene* d_chromosomes;
    int* d_fitness;
    int* d_taskDurations;
    int* d_machineLoads;
    curandState* d_randomStates;
    int populationSize;
    int numTasks;
    int numMachines;
    
    GPUData(int popSize, int tasks, int machines) : 
        populationSize(popSize), numTasks(tasks), numMachines(machines) {
        cudaMalloc(&d_chromosomes, popSize * tasks * sizeof(Gene));
        cudaMalloc(&d_fitness, popSize * sizeof(int));
        cudaMalloc(&d_taskDurations, tasks * sizeof(int));
        cudaMalloc(&d_machineLoads, popSize * machines * sizeof(int));
        cudaMalloc(&d_randomStates, popSize * sizeof(curandState));
    }
    
    ~GPUData() {
        cudaFree(d_chromosomes);
        cudaFree(d_fitness);
        cudaFree(d_taskDurations);
        cudaFree(d_machineLoads);
        cudaFree(d_randomStates);
    }
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
        if (!(file >> taskDurations[i])) {
            cerr << "Error reading task data" << endl;
            exit(1);
        }
    }
    return {numMachines, taskDurations};
}

vector<Gene> greedy(int numMachines, vector<int>& taskDurations) {
    vector<int> taskOrder(taskDurations.size());
    vector<int> machinesLoad(numMachines, 0); 
    vector<Gene> chromosome;

    iota(taskOrder.begin(), taskOrder.end(), 0);

    sort(taskOrder.begin(), taskOrder.end(),
        [&](int a, int b) {
            return taskDurations[a] > taskDurations[b];
        }
    );

    for (int task : taskOrder) {
        int minMachine = min_element(machinesLoad.begin(), machinesLoad.end()) - machinesLoad.begin();
        chromosome.push_back(Gene{task, minMachine});
        machinesLoad[minMachine] += taskDurations[task];
    }

    return chromosome;
}

int fitnessCalculation(int machines, const vector<Gene>& chromosome, const vector<int>& taskDurations) {
    vector<int> timesList(machines, 0);

    for (const auto& gene : chromosome) {
        if (gene.machine < 0 || gene.machine >= machines) {
            cerr << "Invalid machine number: " << gene.machine << endl;
            exit(1);
        }
        if (gene.task < 0 || gene.task >= taskDurations.size()) {
            cerr << "Invalid task number: " << gene.task << endl;
            exit(1);
        }
        timesList[gene.machine] += taskDurations[gene.task];
    }

    return *max_element(timesList.begin(), timesList.end());
}

__global__ void calculateFitnessKernel(Gene* chromosomes, int* fitness, const int* taskDurations, 
                                     int numMachines, int numTasks, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    extern __shared__ int sharedLoads[];
    int* machineLoads = sharedLoads;
    
    for (int i = threadIdx.x; i < numMachines; i += blockDim.x) {
        machineLoads[i] = 0;
    }
    __syncthreads();

    Gene* chromosome = chromosomes + idx * numTasks;
    for (int i = 0; i < numTasks; ++i) {
        int machine = chromosome[i].machine;
        atomicAdd(&machineLoads[machine], taskDurations[chromosome[i].task]);
    }
    __syncthreads();

    int maxLoad = 0;
    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > maxLoad) {
            maxLoad = machineLoads[i];
        }
    }

    fitness[idx] = maxLoad;
}

vector<int> calculateFitnessGPU(GPUData& gpuData, const vector<int>& taskDurations) {
    cudaMemcpy(gpuData.d_taskDurations, taskDurations.data(), 
               taskDurations.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (gpuData.populationSize + blockSize - 1) / blockSize;
    int sharedMemSize = gpuData.numMachines * sizeof(int);

    calculateFitnessKernel<<<gridSize, blockSize, sharedMemSize>>>(
        gpuData.d_chromosomes, gpuData.d_fitness, gpuData.d_taskDurations,
        gpuData.numMachines, gpuData.numTasks, gpuData.populationSize);

    vector<int> fitness(gpuData.populationSize);
    cudaMemcpy(fitness.data(), gpuData.d_fitness, 
               gpuData.populationSize * sizeof(int), cudaMemcpyDeviceToHost);

    return fitness;
}

__global__ void setupRandomStates(curandState* states, unsigned long seed, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void mutationKernel(Gene* chromosomes, int* fitness, const int* taskDurations, 
                             double mutationProbability, double pressure,
                             int numMachines, int numTasks, 
                             int populationSize, curandState* randomStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    curandState* localState = &randomStates[idx];
    Gene* chromosome = chromosomes + idx * numTasks;
    
    int machineLoads[MAX_MACHINES];
    for (int i = 0; i < numMachines; ++i) {
        machineLoads[i] = 0;
    }
    
    for (int i = 0; i < numTasks; ++i) {
        machineLoads[chromosome[i].machine] += taskDurations[chromosome[i].task];
    }
    
    int currentCmax = 0;
    for (int i = 0; i < numMachines; ++i) {
        if (machineLoads[i] > currentCmax) {
            currentCmax = machineLoads[i];
        }
    }

    for (int i = 0; i < numTasks; ++i) {
        Gene& gene = chromosome[i];
        double criticality = (double)machineLoads[gene.machine] / currentCmax;
        double mutationProb = mutationProbability * (1.0 + pressure * criticality);
        
        if (curand_uniform(localState) < mutationProb) {
            int oldMachine = gene.machine;
            int oldLoad = machineLoads[oldMachine];
            
            int bestMachine = -1;
            int minLoad = INT_MAX;
            
            for (int m = 0; m < numMachines; ++m) {
                if (m != oldMachine && machineLoads[m] < minLoad) {
                    minLoad = machineLoads[m];
                    bestMachine = m;
                }
            }
            
            if (bestMachine != -1) {
                int newLoad = machineLoads[bestMachine];
                int updatedOldLoad = oldLoad - taskDurations[gene.task];
                int updatedNewLoad = newLoad + taskDurations[gene.task];
                
                int localOldMax = max(machineLoads[oldMachine], machineLoads[bestMachine]);
                int localNewMax = max(updatedOldLoad, updatedNewLoad);
                
                if (localNewMax <= localOldMax) {
                    machineLoads[oldMachine] = updatedOldLoad;
                    machineLoads[bestMachine] = updatedNewLoad;
                    gene.machine = bestMachine;
                }
            }
        }
    }
}

__global__ void crossoverKernel(Gene* parentChromosomes, Gene* childChromosomes, 
                              const int* taskDurations, int numTasks, 
                              double crossoverRatio, int populationSize, 
                              curandState* randomStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    curandState* localState = &randomStates[idx];
    
    int parent1 = curand(localState) % populationSize;
    int parent2 = curand(localState) % populationSize;
    
    Gene* parent1Chrom = parentChromosomes + parent1 * numTasks;
    Gene* parent2Chrom = parentChromosomes + parent2 * numTasks;
    Gene* childChrom = childChromosomes + idx * numTasks;
    
    double proportion = crossoverRatio;
    int splitPoint = static_cast<int>(numTasks * proportion);
    splitPoint = max(1, min(numTasks - 1, splitPoint));
    
    for (int i = 0; i < splitPoint; ++i) {
        childChrom[i] = parent1Chrom[i];
    }
    
    bool taskPresent[MAX_TASKS] = {false};
    for (int i = 0; i < splitPoint; ++i) {
        taskPresent[childChrom[i].task] = true;
    }
    
    int currentPos = splitPoint;
    for (int i = 0; i < numTasks; ++i) {
        if (!taskPresent[parent2Chrom[i].task]) {
            childChrom[currentPos++] = parent2Chrom[i];
            taskPresent[parent2Chrom[i].task] = true;
        }
    }
    
    for (int i = 0; i < numTasks; ++i) {
        if (!taskPresent[i]) {
            for (int j = 0; j < numTasks; ++j) {
                if (childChrom[j].task == i) continue;
                if (childChrom[j].task == -1) {
                    childChrom[j] = Gene(i, parent1Chrom[i].machine);
                    break;
                }
            }
        }
    }
}

__global__ void initializeRandomChromosomesKernel(Gene* chromosomes, int numTasks, 
                                                int numMachines, int populationSize, 
                                                curandState* randomStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    curandState* localState = &randomStates[idx];
    Gene* chromosome = chromosomes + idx * numTasks;

    for (int i = 0; i < numTasks; ++i) {
        int machine = curand(localState) % numMachines;
        chromosome[i] = Gene(i, machine);
    }
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

pair<vector<vector<Gene>>, vector<int>> initialGeneration(vector<int> taskDurations, 
                                                        int populationsSize, 
                                                        int numMachines, 
                                                        mt19937& gen) {
    vector<vector<Gene>> chromosomes;
    vector<int> fitness;

    auto greedySolution = greedy(numMachines, taskDurations);
    chromosomes.push_back(greedySolution);
    fitness.push_back(fitnessCalculation(numMachines, greedySolution, taskDurations));
    cout << "Greedy Cmax: " << fitness.back() << endl;

    int remainingPopulation = populationsSize - 1;
    if (remainingPopulation <= 0) {
        return make_pair(chromosomes, fitness);
    }

    GPUData gpuData(remainingPopulation, taskDurations.size(), numMachines);

    cudaMemcpy(gpuData.d_taskDurations, taskDurations.data(), 
               taskDurations.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (remainingPopulation + blockSize - 1) / blockSize;
    setupRandomStates<<<gridSize, blockSize>>>(gpuData.d_randomStates, gen(), remainingPopulation);

    initializeRandomChromosomesKernel<<<gridSize, blockSize>>>(
        gpuData.d_chromosomes, taskDurations.size(), numMachines, 
        remainingPopulation, gpuData.d_randomStates);

    auto gpuFitness = calculateFitnessGPU(gpuData, taskDurations);

    vector<Gene> allGenes(remainingPopulation * taskDurations.size());
    cudaMemcpy(allGenes.data(), gpuData.d_chromosomes, 
               allGenes.size() * sizeof(Gene), cudaMemcpyDeviceToHost);

    for (int i = 0; i < remainingPopulation; ++i) {
        chromosomes.emplace_back(
            allGenes.begin() + i * taskDurations.size(),
            allGenes.begin() + (i + 1) * taskDurations.size()
        );
        fitness.push_back(gpuFitness[i]);
    }

    return sortChromosomes(chromosomes, fitness);
}

pair<vector<vector<Gene>>, vector<int>> evolution(vector<vector<Gene>>& chromosomes, 
                                                vector<int>& fitness, 
                                                double mutationProbability, 
                                                int chromosomesPreserved, 
                                                int maxNewChromosomes, 
                                                int numMachines, 
                                                vector<int>& taskDurations, 
                                                double crossoverRatio, 
                                                double pressure, 
                                                mt19937& gen) {
    vector<vector<Gene>> newPopulation(chromosomes.begin(), chromosomes.begin() + chromosomesPreserved);
    
    GPUData gpuData(maxNewChromosomes, taskDurations.size(), numMachines);
    
    vector<Gene> allParentGenes;
    for (const auto& chrom : chromosomes) {
        allParentGenes.insert(allParentGenes.end(), chrom.begin(), chrom.end());
    }
    cudaMemcpy(gpuData.d_chromosomes, allParentGenes.data(), 
               allParentGenes.size() * sizeof(Gene), cudaMemcpyHostToDevice);
    
    cudaMemcpy(gpuData.d_taskDurations, taskDurations.data(), 
               taskDurations.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (gpuData.populationSize + blockSize - 1) / blockSize;
    setupRandomStates<<<gridSize, blockSize>>>(gpuData.d_randomStates, gen(), gpuData.populationSize);
    
    Gene* d_childChromosomes;
    cudaMalloc(&d_childChromosomes, maxNewChromosomes * taskDurations.size() * sizeof(Gene));
    
    crossoverKernel<<<gridSize, blockSize>>>(
        gpuData.d_chromosomes, d_childChromosomes, gpuData.d_taskDurations,
        taskDurations.size(), crossoverRatio, chromosomes.size(), gpuData.d_randomStates);
    
    mutationKernel<<<gridSize, blockSize>>>(
        d_childChromosomes, gpuData.d_fitness, gpuData.d_taskDurations,
        mutationProbability, pressure, numMachines, taskDurations.size(),
        maxNewChromosomes, gpuData.d_randomStates);
    
    vector<Gene> allChildGenes(maxNewChromosomes * taskDurations.size());
    cudaMemcpy(allChildGenes.data(), d_childChromosomes, 
               allChildGenes.size() * sizeof(Gene), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < maxNewChromosomes; ++i) {
        newPopulation.emplace_back(
            allChildGenes.begin() + i * taskDurations.size(),
            allChildGenes.begin() + (i + 1) * taskDurations.size()
        );
    }
    
    vector<int> newFitness;
    for (const auto& chromosome : newPopulation) {
        newFitness.push_back(fitnessCalculation(numMachines, chromosome, taskDurations));
    }
    
    auto result = sortChromosomes(newPopulation, newFitness);
    
    cudaFree(d_childChromosomes);
    
    return result;
}

int main() {
    random_device rd;
    mt19937 gen(rd());

    Config config = loadConfig("config.json");

    auto parseResult = parseData(config.dataFile);
    int numMachines = parseResult.first;
    vector<int> taskDurations = parseResult.second;
    if (numMachines <= 0 || taskDurations.empty()) {
        cerr << "Invalid input data" << endl;
        return 1;
    }

    int chromosomesPreserved = max(1, static_cast<int>(config.populationsSize * config.chromosomesPreservedPercentage / 100.0));
    int maxNewChromosomes = config.populationsSize - chromosomesPreserved;

    auto initGenResult = initialGeneration(taskDurations, config.populationsSize, numMachines, gen);
    vector<vector<Gene>> chromosomes = initGenResult.first;
    vector<int> fitness = initGenResult.second;
    BestChromosome bestChromosome = BestChromosome(chromosomes[0], fitness[0], 1);

    int totalTaskTime = accumulate(taskDurations.begin(), taskDurations.end(), 0);
    int lowerBound = static_cast<int>(ceil(static_cast<double>(totalTaskTime) / numMachines));

    cout << "Initial best Cmax: " << bestChromosome.fitness << endl;
    cout << "Lower bound: " << lowerBound << endl;
    cout << "Starting evolution..." << endl;

    auto startTime = chrono::steady_clock::now();

    for (int generation = 1; generation <= config.generations; ++generation) {
        auto currentTime = chrono::steady_clock::now();
        chrono::duration<double> elapsedTime = currentTime - startTime;
        if (elapsedTime.count() >= config.maxTime) {
            cout << "\nTime limit reached!" << endl;
            break;
        }

        if (generation % 10000 == 0) {
            cout << "Generation: " << generation 
                 << " Best Cmax: " << bestChromosome.fitness 
                 << " Time: " << elapsedTime.count() << "s" << endl;
        }

        tie(chromosomes, fitness) = evolution(
            chromosomes, fitness, config.mutationProbability, 
            chromosomesPreserved, maxNewChromosomes, numMachines, 
            taskDurations, config.crossoverRatio, config.mutationPressure, gen);

        if (fitness[0] < bestChromosome.fitness) {
            bestChromosome = {chromosomes[0], fitness[0], generation};
            cout << "Generation " << generation << ": New best Cmax = " << bestChromosome.fitness 
                 << " Elapsed time: " << elapsedTime.count() << "s" << endl;
        }
    }

    cout << "\nFinal Results:" << endl;
    cout << "Best Cmax: " << bestChromosome.fitness << endl;
    cout << "Found in generation: " << bestChromosome.generation << endl;
    cout << "Lower bound: " << lowerBound << endl;
    cout << "Gap to lower bound: " 
         << (static_cast<double>(bestChromosome.fitness - lowerBound) / lowerBound * 100.0) 
         << "%" << endl;

    auto endTime = chrono::steady_clock::now();
    chrono::duration<double> totalTime = endTime - startTime;
    cout << "Total execution time: " << totalTime.count() << " seconds" << endl;

    return 0;
}
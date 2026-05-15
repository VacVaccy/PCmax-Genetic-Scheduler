#include "utils.hpp"
#include "Crossing.hpp"
#include "DataParser.hpp"
#include "Fitness.hpp"
#include "GreedyScheduler.hpp"
#include "Mutation.hpp"

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
#ifndef UTILS_HPP
#define UTILS_HPP

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

#endif
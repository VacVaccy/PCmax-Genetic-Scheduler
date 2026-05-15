#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "utils.hpp"

// vector<Gene> mutation(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange);
void mutation_inplace(double mutationRate, vector<Gene>& chromosome, int numMachines, int mutationRange);

#endif

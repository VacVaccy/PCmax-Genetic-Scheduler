#include "utils.hpp"

int fitnessCalculation(int machines, vector<Gene>& chromosome, vector<int>& tasks) {
    vector<int> timesList(machines, 0);

    for (int i = 0; i < chromosome.size(); i++) {
        int index = chromosome[i].machine - 1;
        timesList[index] += chromosome[i].task;
    }

    int Cmax = *max_element(timesList.begin(), timesList.end());

    return Cmax;
}
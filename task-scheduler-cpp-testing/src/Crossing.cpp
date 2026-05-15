#include "utils.hpp"

pair<vector<Gene>, vector<Gene>> crossing(const vector<Gene>& chromosome1, const vector<Gene>& chromosome2, double proportion) {
    proportion = max(0.1, min(0.9, proportion));
    int splitPoint = static_cast<int>(chromosome1.size() * proportion);
    vector<Gene> newChromosome1 = vector<Gene>(chromosome1.begin(), chromosome1.begin() + splitPoint);
    newChromosome1.insert(newChromosome1.end(), chromosome2.begin() + splitPoint, chromosome2.end());

    vector<Gene> newChromosome2 = vector<Gene>(chromosome2.begin(), chromosome2.begin() + splitPoint);
    newChromosome2.insert(newChromosome2.end(), chromosome1.begin() + splitPoint, chromosome1.end());
    return {newChromosome1, newChromosome2};
}
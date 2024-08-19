#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "global.h"
#include "instance.h"
#include "solution.h"
using namespace std;

class Solver {
public:
    Solution *population[POPULATION_SIZE];
    Solution *best_solution;
    Instance instance;
    vector<std::vector<int>> neighborhood_;
    std::vector<std::vector<double>> lambda_;
    std::vector<double> z_;
    int iteration;
    int evaluation;

    Solver(const Instance& instance): instance(instance){
        iteration = 0;
        evaluation = 0;
        for(int i = 0; i < POPULATION_SIZE; i++){
            population[i] = new Solution(instance);
        }
        best_solution = new Solution(instance);
        std::cout << std::fixed << std::setprecision(5);
    }
    void deepCopyBestSol(Solution* minSolution);
    void writeFile(const std::string& fileName, const std::string& fileContent);
    void init();
    void diversity_enhance();
    void initUniformWeight();
    void initNeighborhood();
    vector<int> getRandomPermutation(int n);
    void updateReference(Solution* individual);
    double distVector(const std::vector<double>& vector1, const std::vector<double>& vector2);
    void  minFastSort(std::vector<double>& x, std::vector<int>& idx, int n, int m);
    void initPopulation();
    void initIdealPoint();
    double randDouble();
    int randInt(int min, int max);
    void matingSelection(std::vector<int>& list, int cid, int size, int type);
    void Crossover(Solution* parent0, Solution* parent1, Solution* child);
    void Mutation(Solution* child, int evaluation);
    void positionBasedCrossover(Solution* parent1, Solution* parent2);
    void updateProblem(Solution* child, int id, int type);
    double fitnessFunction(const Solution* solution, const std::vector<double>& lambda);

    void print_pop();
    vector<pair<int,int> > nBestSelection(int nBest);
    vector<int> order_crossover(Solution *parent1, Solution *parent2);
    void memetic_solve();
    void solve(int max_evaluations, int benchmark_id);
    vector<int> random_selection(int size);
    int num_of_same_solutions();
    unordered_map<double,int> diversityMaintain(int customerSize);
    vector<int> binary_tournament();
    vector<int> parent_select(int type);
};
#endif 

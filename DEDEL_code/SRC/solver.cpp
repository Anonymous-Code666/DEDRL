#include "solver.h"
#include <algorithm>
#include <map>
#include <cassert>
#include <cmath>

void Solver::init() {
    // STEP 1. Initialization
    // STEP 1.1. Compute euclidean distances between weight vectors and find T
    initUniformWeight();
    initNeighborhood();
    // STEP 1.2. Initialize population
    initPopulation();
    // STEP 1.3. Initialize z_
    initIdealPoint();
}
//Compute euclidean distances between weight vectors and find T
void Solver::initUniformWeight(){
    lambda_ = std::vector<std::vector<double>>(POPULATION_SIZE, std::vector<double>(num_of_objective));

    for (int n = 0; n < POPULATION_SIZE; n++) {
        double a = 1.0 * n / (POPULATION_SIZE - 1);
        lambda_[n][0] = a; // distance_cost
        lambda_[n][1] = 1 - a; // vehilcle_cost
    }
}

void Solver::initNeighborhood(){
    neighborhood_ = std::vector<std::vector<int>>(POPULATION_SIZE, std::vector<int>(T_));
    std::vector<double> x(POPULATION_SIZE);
    std::vector<int> idx(POPULATION_SIZE);

    for (int i = 0; i < POPULATION_SIZE; i++) {
        for (int j = 0; j < POPULATION_SIZE; j++) {
            x[j] = distVector(lambda_[i], lambda_[j]);
            idx[j] = j;
        }

        minFastSort(x, idx, POPULATION_SIZE, T_);

        std::copy(idx.begin(), idx.begin() + T_, neighborhood_[i].begin());
    }
}

void Solver::initPopulation(){
    for(int i = 0; i < POPULATION_SIZE; ++i) {
        population[i]->get_random_permutation();
         population[i]->construct_solution_v2();
    }
    sort(population, population + POPULATION_SIZE, [](Solution *a, Solution *b) {
        return a->total_cost < b->total_cost;
    });
    Solution* minSolution = *std::min_element(population, population + POPULATION_SIZE, [](const Solution* a, const Solution* b) { return a->total_cost < b->total_cost; });
    deepCopyBestSol(minSolution);
    std::stringstream ss;
    ss <<  "gen: "<< evaluation/SLOT<<",the best cost is: " << best_solution->total_cost << endl;
    std::string res = ss.str();
    writeFile(convergence_file, res);
}

void Solver::initIdealPoint(){
    z_ = std::vector<double>(num_of_objective, std::numeric_limits<double>::max());

    for (int ind = 0; ind < POPULATION_SIZE; ind++) {
        updateReference(population[ind]);
    }
}

void Solver::updateReference(Solution* individual) {
    //z_[0]:distance_cost, z_[1]: vehilcle_cost
    if (individual->distance_cost < z_[0]) {
        z_[0] = individual->distance_cost;
    }
    if (individual->vehilcle_cost < z_[1]) {
        z_[1] = individual->vehilcle_cost;
    }
}

double Solver::distVector(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int dim = vector1.size();
    double sum = 0;
    for (int n = 0; n < dim; n++) {
        sum += (vector1[n] - vector2[n]) * (vector1[n] - vector2[n]);
    }
    return std::sqrt(sum);
}

void Solver::minFastSort(std::vector<double>& x, std::vector<int>& idx, int n, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < n; j++) {
            if (x[i] > x[j]) {
                std::swap(x[i], x[j]);
                std::swap(idx[i], idx[j]);
            }
        }
    }
}

// Perform deep copy to best solution
void Solver::deepCopyBestSol(Solution* minSolution) {
    best_solution->undelmiter_planned_route = minSolution->undelmiter_planned_route;
    best_solution->routes = minSolution->routes;
    best_solution->total_cost = minSolution->total_cost;
    best_solution->v_num = minSolution->v_num;
    best_solution->vehilcle_cost = minSolution->vehilcle_cost;
    best_solution->distance = minSolution->distance;
    best_solution->distance_cost = minSolution->distance_cost;
}


// Function to write content to a file
void Solver::writeFile(const std::string& fileName, const std::string& fileContent) {
    std::ofstream outFile(fileName, std::ios::app);
    if (outFile.is_open()) {
        outFile << fileContent;
        outFile.close();
    } else {
        std::cerr << "Unable to open file '" << fileName << "' for writing." << std::endl;
    }
}


struct ParentPair {
    int parent1;
    int parent2;
};

vector<ParentPair> generateParentPairs() {
    vector<ParentPair> parentPairs;

    random_device rd;
    mt19937 gen(rd());

    vector<int> bestHalfIndices, worstHalfIndices;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (i < POPULATION_SIZE / 2) {
            bestHalfIndices.push_back(i);
        } else {
            worstHalfIndices.push_back(i);
        }
    }
    shuffle(bestHalfIndices.begin(), bestHalfIndices.end(), gen);
    shuffle(worstHalfIndices.begin(), worstHalfIndices.end(), gen);

    for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
        ParentPair pair = {bestHalfIndices[i], worstHalfIndices[i]};
        parentPairs.push_back(pair);
    }

    shuffle(bestHalfIndices.begin(), bestHalfIndices.end(), gen);
    shuffle(worstHalfIndices.begin(), worstHalfIndices.end(), gen);
    for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
        ParentPair pair = {bestHalfIndices[i], worstHalfIndices[i]};
        parentPairs.push_back(pair);
    }

    return parentPairs;
}

vector<int> Solver::parent_select(int type) {
    int k1, k2,sel1,sel2;
    srand(time(NULL));
    sel1 = rand() % (POPULATION_SIZE/2);
    if(type==1){
        do{
            sel2=rand() % (POPULATION_SIZE/2);
        }while (sel2 == sel1);
    } else{
        sel2 = rand() % (POPULATION_SIZE/2)+POPULATION_SIZE/2;
    }
    vector<int> select_parents;
    select_parents.push_back(sel1);
    select_parents.push_back(sel2);
    return select_parents;
}

vector<int> Solver::binary_tournament() {
    int k1, k2,sel1,sel2;
    srand(time(NULL));
    k1 = rand() % (POPULATION_SIZE/2);
    sel1 = k1;

    k2 = rand() % (POPULATION_SIZE/2)+POPULATION_SIZE/2;
    sel2 = k2;
    vector<int> select_parents;
    select_parents.push_back(sel1);
    select_parents.push_back(sel2);
    return select_parents;
}

void Solver::Crossover(Solution *parent1,  Solution *parent2,  Solution *child) {
    child->undelmiter_planned_route = order_crossover(parent1,parent2);
    child->construct_solution_v2();
}

void Solver::Mutation(Solution *child, int evaluation) {
    double perturb_ratio = std::max(initial_perturb_ratio*std::pow(decayRate, evaluation / (double)max_evaluations), 1.0);
    vector<vector<int>> perturb_routes;
    int try_mutation=0;
    std::vector<int> temp_undelmiter_route = child->undelmiter_planned_route;
    double tc0 = child->total_cost;
    do{
        if(try_mutation>0) child->undelmiter_planned_route = temp_undelmiter_route;
        child->mutation();
        try_mutation += 1;
    }while(child->total_cost > tc0*perturb_ratio);
}

void Solver::diversity_enhance() {
    cout<<"reinitialize the later half population"<<endl;
    for(int i = NC; i < POPULATION_SIZE; i++){
        population[i]->get_random_permutation();
        population[i]->construct_solution_v2();
    }
}

void Solver::solve(int max_evaluations, int benchmark_id) {
    cout <<"best solution: "<< best_solution->v_num << " "<< best_solution->distance_cost  <<" " << best_solution->total_cost << endl;

    int no_improve = 0;
    double best_cost = best_solution->total_cost;
    while(evaluation < max_evaluations) {
        vector<int> permutation = getRandomPermutation(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            int n = permutation[i]-1;
            double rnd = randDouble();
            int type;
            if (rnd < delta_) {
                type = 1; // neighborhood
            } else {
                type = 2; // whole population
            }
            // vector<int> p = binary_tournament();
            std::vector<int> p;
            if(randDouble()<sel_elite){
                p = parent_select(1);
            }else{
                p = parent_select(2);
            }
            Solution *child = new Solution(instance);
            Solution *parent1 = population[p[0]];
            Solution *parent2 = population[p[1]];
            //Crossover
            Crossover(parent1,parent2, child);
            //Mutation
            double mp = randDouble();
            if(mp <= MP) {
                Mutation(child, evaluation);
            }
            //DQN Search
            child->dqnSearch();
            child->evaluate();
            evaluation++;
            // STEP 2.4. Update z_
            updateReference(child);
            // STEP 2.5. Update of solutions
            updateProblem(child, n, type);
            delete child;//avoid OOM
        }
        sort(population, population + POPULATION_SIZE, [](Solution *a, Solution *b) {
            return a->total_cost < b->total_cost;
        });
        Solution* minSolution = *std::min_element(population, population + POPULATION_SIZE, [](const Solution* a, const Solution* b) { return a->total_cost < b->total_cost; });
        if(minSolution->total_cost < best_cost) {
            best_cost = minSolution->total_cost;
            deepCopyBestSol(minSolution);
            no_improve = 0;
        }else {
            no_improve ++;
        }
        if(no_improve==0) cout << "evaluation: "<< evaluation<<",the best solution is: "<<best_solution->v_num<< " "<< best_solution->distance_cost<<" " << best_solution->total_cost << endl;
        if (evaluation % SLOT == 0) {
            std::stringstream ss;
            ss <<  "gen: "<< evaluation/POPULATION_SIZE<<",the best cost is: " <<best_solution->v_num<< " "<< best_solution->distance_cost<<" " << best_solution->total_cost << endl;
            std::string res = ss.str();
            writeFile(convergence_file, res);
        }
    }
}

void Solver::
updateProblem(Solution* child, int id, int type) {
    int size;
    int time = 0;

    if (type == 1) {
        size = neighborhood_[id].size();
    } else {
        size = POPULATION_SIZE;
    }

    std::vector<int> perm = getRandomPermutation(size);

    for (int i = 0; i < size; ++i) {
        int k;
        if (type == 1) {
            k = neighborhood_[id][perm[i]-1];
        } else {
            k = perm[i]-1;
        }

        double f1 = fitnessFunction(population[k], lambda_[k]);
        double f2 = fitnessFunction(child, lambda_[k]);

        if (f2 < f1 && !(population[k]->total_cost<=best_solution->total_cost && child->total_cost>=population[k]->total_cost)) {
            population[k]->deepCopy(child);
            time++;
        }

        if (time >= nr_) {
            break;
        }
    }
}

double Solver::fitnessFunction(const Solution* solution, const std::vector<double>& lambda) {
    double fitness = 0.0;

    if (functionType_ == "_TCHE1") {
        double maxFun = -1.0e+30;
        std::vector<double> diff;
        diff.push_back(std::abs(solution->distance_cost - z_[0]));
        diff.push_back(std::abs(solution->vehilcle_cost - z_[1]));

        for (int n = 0; n < num_of_objective; ++n) {
            double feval;
            if (lambda[n] == 0) {
                feval = 0.0001 * diff[n];
            } else {
                feval = diff[n] * lambda[n];
            }
            if (feval > maxFun) {
                maxFun = feval;
            }
        }
        fitness = maxFun;
    } else {
        std::cerr << "MOEAD.fitnessFunction: unknown type " << functionType_ << std::endl;
        std::exit(-1);
    }
    return fitness;
}



void Solver::matingSelection(std::vector<int>& list, int cid, int size, int type) {
    int ss;
    int r;
    int p;

    ss = neighborhood_[cid].size();
    while (list.size() < static_cast<size_t>(size)) {
        if (type == 1) {
            r = randInt(0, ss - 1);
            p = neighborhood_[cid][r];
        } else {
            p = randInt(0, POPULATION_SIZE - 1);
        }

        bool flag = true;
        for (int i = 0; i < list.size(); i++) {
            if (list[i] == p) { 
                flag = false;
                break;
            }
        }

        if (flag) {
            list.push_back(p);
        }
    }
}

int Solver::randInt(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

double Solver::randDouble(){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(0, 1);
    double random_number = dis(gen);
    return  random_number;
}

vector<int>  Solver::getRandomPermutation(int n){
    random_device rd;
    mt19937 gen(rd());
    std::vector<int> perm(n);
    for(int i = 0; i < n; ++i) {
        perm[i] = i + 1;
    }
    shuffle(perm.begin(), perm.end(), gen);
    return perm;
}

 void Solver::memetic_solve() {
    int no_improve = 0;
    vector<pair<int,int>> eliteGroup = nBestSelection(n_best);
    double best_value = population[0]->total_cost;

    while(evaluation < max_evaluations) {
        for(int i = 0; i < POPULATION_SIZE; ++i) {
            pair<int,int> p;
            if(no_improve < max_no_imp) {
                p = eliteGroup[i];
            }else {
                p.first = 0;
                p.second = random_selection(1)[0];
            }
            Solution *parent1 = new Solution(*population[p.first]);
            Solution *parent2 = new Solution(*population[p.second]);
            Solution *child = new Solution(instance);
            child->undelmiter_planned_route = order_crossover(parent1,parent2);
            child->check_feasible();
            child->dqnSearch();
            child->evaluate();
            evaluation++;
            cout << "evaluation: "<< evaluation<<",the child cost is: " << child->total_cost << endl;
            sort(population, population + POPULATION_SIZE + 1, [](Solution *a, Solution *b) {
                return a->total_cost < b->total_cost;
            });

            cout << "the best cost is : " << population[0]->total_cost << endl;
            if(population[0]->total_cost < best_value) {
                best_value = population[0]->total_cost;
                best_solution = population[0];
                no_improve = 0;
            }else {
                no_improve ++;
            }
            int same_num = num_of_same_solutions();
            diversityMaintain(instance.customerSize);

            if(evaluation >= max_evaluations) {
                break;
            }
        }
    }
}

int rand_choose(int num)
{
    int k = rand()%num;
    return k;
}

vector<int> Solver::order_crossover(Solution* parent1, Solution* parent2) {
    const std::vector<int>& route1 = parent1->undelmiter_planned_route;
    const std::vector<int>& route2 = parent2->undelmiter_planned_route;

    assert(route1.size() == route2.size());
    int routeSize = (int)route1.size();

    std::vector<int> crossoverPoints(2);
    std::vector<int> segment1;

    int length =routeSize-1;
    int length1=length/2;
    int a = rand_choose(length);
    int b = (a+1+rand_choose(length1))%length;

    if (a > b)
    {
        int tmp = a;
        a = b;
        b = tmp;
    }
    if((b-a+1)==length)a++;
    crossoverPoints[0] = a;
    crossoverPoints[1] = b;



    std::vector<int> childUndelimiterPlannedRoute(routeSize, -1);

    for (int i = crossoverPoints[0]; i <= crossoverPoints[1]; ++i) {
        segment1.push_back(route1[i]);
        childUndelimiterPlannedRoute[i] = route1[i];
    }




    int idx = 0;
    for (int i = 0; i < routeSize; ++i) {
        if (childUndelimiterPlannedRoute[i] == -1) {
            while (std::find(segment1.begin(), segment1.end(), route2[idx]) != segment1.end()) {
                ++idx;
            }
            childUndelimiterPlannedRoute[i] = route2[idx];
            ++idx;
        }
    }
    //If the sequence of the child is same to parent, random reverse
    bool isSame1 = true,isSame2=true;
    for (int i = 0; i <childUndelimiterPlannedRoute.size(); ++i) {
        if(childUndelimiterPlannedRoute[i] != route1[i]) {
            isSame1 = false;
            break;
        }
    }
    for (int i = 0; i <childUndelimiterPlannedRoute.size(); ++i) {
        if(childUndelimiterPlannedRoute[i] != route2[i]) {
            isSame2 = false;
            break;
        }
    }
    if(isSame1 || isSame2) {
        // Random reversion
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, childUndelimiterPlannedRoute.size() - 2);
        int random_s = dis(gen);
        std::uniform_int_distribution<> dis2(1, childUndelimiterPlannedRoute.size()/2);
        int random_t = random_s + dis2(gen);
        while (random_t >= childUndelimiterPlannedRoute.size()){
            random_t =  random_s + dis2(gen);
        }
        //reverse the sequence between random_s and random_t
        std::reverse(childUndelimiterPlannedRoute.begin() + random_s, childUndelimiterPlannedRoute.begin() + random_t+1);
    }
    return childUndelimiterPlannedRoute;
}


void Solver::print_pop() {
    for(int i = 0; i < POPULATION_SIZE; ++i) {
        cout << population[i]->total_cost << endl;
        population[i]->print_undelmiter_routes();
    }
}


vector<int> Solver::random_selection(int size) {
    vector<int> selected_parents;
    while(selected_parents.size() < size) {
        int p = rand() % (POPULATION_SIZE - n_best) + n_best;
        if(find(selected_parents.begin(),selected_parents.end(),p) == selected_parents.end()) {
            selected_parents.push_back(p);
        }
    }
    return selected_parents;
}

vector<pair<int,int>> Solver::nBestSelection(int nBest) {
    vector<pair<int,int>> matingGroup;
    int divSize = POPULATION_SIZE / nBest;
    int cnt = 0;

    for(int i = 0; i < n_best; ++i) {
        for(int j = 0; j < divSize; ++j) {
            int idx1 = i;
            int idx2 = i + j + 1;
            matingGroup.push_back(make_pair(idx1,idx2));
            cnt++;
        }
    }
    return matingGroup;
}

int Solver::num_of_same_solutions() {
    std::map<double,int> fitness_counts;
    int max_count = 0;

    for(auto sol:population) {
        int count = fitness_counts[sol->total_cost] + 1;
        fitness_counts[sol->total_cost] = count;

        if(count > max_count) {
            max_count = count;
        }
    }

    return max_count;
}

unordered_map<double,int> Solver::diversityMaintain(int customerSize) {
    unordered_map<double,int> diversity;
    for(int i = 0; i < POPULATION_SIZE; ++i) {
        Solution *sol = population[i];
        double fitness = sol->total_cost;
        int count = diversity[fitness] + 1;
        diversity[fitness] = count;

        if(count > 1) {
            population[i]->get_random_permutation();
            population[i]->construct_solution();
            population[i]->evaluate();
        }
    }
    return diversity;
}


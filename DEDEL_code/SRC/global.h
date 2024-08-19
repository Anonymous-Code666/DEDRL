
#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>
using namespace std;

const int POPULATION_SIZE = 36;
const int runs = 2;
const int MAX_RUN = 2;
const int generations = 500;
const double dis_factor = 0.014;
const double vehicle_factor = 300;
const int max_evaluations = 18000;
const int max_env_steps = 5;
const int max_no_imp = 3;
const int NO_IMP = 25;
const int print_slot=180;
const double MP = 0.3; //mutation probability
const int customer_sizes_arr[] = {200, 200, 200, 200, 400, 400, 400, 400, 600, 600, 600, 600, 800, 800, 800, 800, 1000, 1000, 1000, 1000};
const string problem_dirs[] = {
    "./benchmark/200_1.vrpsdptw", "./benchmark/200_2.vrpsdptw", "./benchmark/200_3.vrpsdptw", "./benchmark/200_4.vrpsdptw",
    "./benchmark/400_1.vrpsdptw", "./benchmark/400_2.vrpsdptw", "./benchmark/400_3.vrpsdptw", "./benchmark/400_4.vrpsdptw",
    "./benchmark/600_1.vrpsdptw", "./benchmark/600_2.vrpsdptw", "./benchmark/600_3.vrpsdptw", "./benchmark/600_4.vrpsdptw",
    "./benchmark/800_1.vrpsdptw", "./benchmark/800_2.vrpsdptw", "./benchmark/800_3.vrpsdptw", "./benchmark/800_4.vrpsdptw",
    "./benchmark/1000_1.vrpsdptw", "./benchmark/1000_2.vrpsdptw", "./benchmark/1000_3.vrpsdptw", "./benchmark/1000_4.vrpsdptw"
};
const double max_capacity = 2.5;
const int MAX_D = 1000;
const int num_of_objective = 2;
const int T_ = 12;
const int nr_ = 1;
const double delta_ = 0.9;
const double sel_elite=0.75;
const int evaluations_of_stage = 1800;
const double diversity_portion = 0.8;
const int n_best = 4;
const double DOUBLE_INF = 99999999.0;
const int MAX_NO_IMP=1;
const string result_file="result.txt";
//const string convergence_file="convergence.txt";
const int SLOT=36;
const string functionType_ = "_TCHE1";
//static double perturb_ratio=1;
const double initial_perturb_ratio=1.1;
const double decayRate=0.9;
const double ls_ratio=1;
const double perturb_max_k_ratio=1;
const double ls_max_k_ratio=0.5;
//2-opt parameter set
static double dqn_total_imp=0;//used to determinate the reward of 2-opt
static double two_opt_imp=0;//the improved cost of one 2-opt
const int reverseLen=5;
const int NC = 18;
const int NP = 5;
const int INTERVAL=1800;
const int COST_SPACE = 0.5;
//bool accept_perturb=false;
extern int problem_idx;
//extern string cur_time;
extern string convergence_file;
#endif //GLOBAL_H

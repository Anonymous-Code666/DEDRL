#include <iostream>
#include "global.h"
#include "instance.h"
#include "solution.h"
#include "solver.h"
using namespace std;
std::ofstream outputFile;
int problem_idx;
string convergence_file;
int main() {
    std::cout << std::fixed << std::setprecision(5);
    for(int run=0;run<MAX_RUN;run++){
        for(int benchmark_id = 0; benchmark_id < 20; benchmark_id++) {
            int customer_size = customer_sizes_arr[benchmark_id];
            problem_idx=benchmark_id;
            string problem_dir = problem_dirs[benchmark_id];
            cout << problem_dir << endl;
            std::time_t t = std::time(nullptr); 
            char buffer[80];
            std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", std::localtime(&t)); 
            std::string str(buffer);
            convergence_file="convergence_v5.5_"+std::to_string(benchmark_id)+"_"+str+".txt";
            std::streambuf *coutbuf = std::cout.rdbuf(); 
            outputFile.open("search_log_v5.5_"+std::to_string(benchmark_id)+"_"+str+".txt", std::ios::app);
            std::cout.rdbuf(outputFile.rdbuf()); 
            cout<< "------------benchmark-------------"<< benchmark_id<<endl;
            int max_evaluations=18000;
            Instance instance(customer_size);
            instance.readInputFile(problem_dir);
            Solver solver(instance);
            solver.writeFile(convergence_file, "run_"+std::to_string(run)+" ,benchmark_id "+problem_dirs[benchmark_id] + "\n");
            auto start_time = std::chrono::high_resolution_clock::now();
            solver.init();
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> used_time = end_time - start_time;

            solver.solve(max_evaluations, benchmark_id);
            end_time = std::chrono::high_resolution_clock::now();
            used_time = end_time - start_time;
            std::stringstream ss;
            ss << "run_"<< run <<" ,benchmark_id: " << benchmark_id << "\n" << "v_num, distance_cost, Total_cost, used_time:\n "
            << solver.best_solution->v_num<<", "<< solver.best_solution->distance_cost<<", "<< solver.best_solution->total_cost<<", "<< used_time.count()<< endl;
            std::string res = ss.str();
            solver.writeFile(result_file, res);
            cout << res << endl;
            std::cout.rdbuf(coutbuf);
            outputFile.close();
        }
    }

}

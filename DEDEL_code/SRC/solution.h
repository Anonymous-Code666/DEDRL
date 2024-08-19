#ifndef SOLUTION_H
#define SOLUTION_H

#include <vector>
#include <algorithm>
#include <random>

#include "instance.h"
#include "global.h"

#include <torch/torch.h>

class Solution {
public:
    Instance instance;
    std::vector<int> undelmiter_planned_route;
    std::vector<std::vector<int>> routes;
    double total_cost;
    double v_num;
    double vehilcle_cost;
    double distance;
    double distance_cost;

    Solution(const Instance& instance): instance(instance){
        total_cost = DOUBLE_INF;
        v_num = DOUBLE_INF;
        vehilcle_cost = DOUBLE_INF;
        distance = DOUBLE_INF;
        distance_cost = DOUBLE_INF;
    }


    void print_undelmiter_routes();
    void print_delmiter_routes();
    void get_random_permutation();

    bool check_feasible();

    bool check_time_window(vector<int>& route);
    bool check_capacity(vector<int>& route);
    bool is_violate_constraint(vector<int>& route);

    double get_route_dis(vector<int> &route);
    double get_total_dis();
    double get_weight_delta_distance(std::vector<int> &route, int insert_pos);

    void construct_solution();
    void construct_solution_v2();

    void get_undel_routes(std::vector<std::vector<int> > &routes);
    void twoOptPerturbationIter();
    vector<vector<int>> perturbation(double perturb_ratio);
    void twoOptSwap(std::vector<int>& tour, int i, int k);
    std::vector<std::vector<int>> decode_split_perturb(std::vector<int>& temp_undelmiter_route, double scale_factor, int max_k_size);
    std::vector<std::vector<int>> decode_split_2opt(std::vector<int>& temp_undelmiter_route, double scale_factor, int max_k_size);
    void mutation();
    void decode_split(std::vector<int>& temp_undelmiter_route, int max_k_size);
    void deepCopy(Solution* other);
    double evaluate();
    struct Experience {
        torch::Tensor state;
        int action;
        double reward;
        torch::Tensor next_state;
        bool done;
    };

    class VRPEnv {
    public:
        VRPEnv(Solution &sln);
        torch::Tensor toTensor(Solution &solution);
        torch::Tensor reset();
        std::tuple<torch::Tensor, double, bool> step(int action);
        Solution &getSolution() { return sln_; }
        int env_step_count_;

    private:
        Solution &initSln_;
        Solution &sln_;
        torch::Tensor initState_;
        torch::Tensor state_;
        int step_no_improve;
        int maxNoImprove;
    };

    class ReplayMemory {
       public:
        explicit ReplayMemory(int capacity);
        void push(const torch::Tensor &state, const int action, const double reward,
                  const torch::Tensor &next_state, const bool done);
        std::vector<Experience> sample(int batch_size);
        int size() const;

       private:
        std::deque<Experience> memory_;
        int capacity_;
        std::mt19937 generator_;
    };

    struct QNet : torch::nn::Module {
        QNet(int num_inputs, int num_actions, int hidden_size)
            : fc1_(register_module("fc1_",
                                   torch::nn::Linear(num_inputs, hidden_size))),
              fc2_(register_module("fc2_",
                                   torch::nn::Linear(hidden_size, num_actions))) {}

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc1_->forward(x));
            x = fc2_->forward(x);
            return x;
        }

        torch::nn::Linear fc1_, fc2_;

        void save_weights(const std::string& file_path) {
            torch::serialize::OutputArchive output_archive;
            this->save(output_archive);
            output_archive.save_to(file_path);
        }

        void load_weights(const std::string& file_path) {
            torch::serialize::InputArchive input_archive;
            input_archive.load_from(file_path);
            this->load(input_archive);
        }
    };

    class Agent {
       public:
        Agent();
        Agent(Agent &agent);
        Agent(int num_inputs, int num_actions, int hidden_size,
              double epsilon_start, double epsilon_end, int num_steps,
              int update_target, double gamma, int batch_size, int memory_capacity,
              double tau,Solution &sln,
              std::string qNetFilePath,std::string qTargetFilePath);

        void act(torch::Tensor &state, int &action, double &reward,
                 torch::Tensor &next_state, bool &done);

        void learn();

        void train(int kNumEpisodes,std::string q_net_path,std::string q_target_path);


       private:
        VRPEnv env_;
        QNet q_net_, q_target_;
        ReplayMemory replay_memory_;
        torch::optim::Adam optimizer_;
        int num_actions_;
        std::function<double(int)> epsilon_by_step_;
        int batch_size_, update_target_, num_steps_, step_count_, episode_count_;
        int max_no_improve_;
        double gamma_;
        double tau_;
        int num_inputs_;
        int hidden_size_;
        double epsilon_start_;
        double epsilon_end_;
    };

    void dqnSearch();
    double intra_exchange();
    double inter_relocate();

    double best_inter_exchange();
    double best_intra_exchange();
    double best_inter_relocate();
    double best_intra_relocate();
    double twoOpt();
    double two_opt();
    double best_2_opt_star();
    double best_or_opt();
    double best_CROSS_exchange();

    std::tuple<int,int,double> relocate(int v_id, int cus_idx,vector<vector<int> > plan_routes);

    void delete_delimiter();
};


#endif 

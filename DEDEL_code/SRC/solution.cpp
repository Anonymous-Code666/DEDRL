#include "solution.h"
#include <set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "global.h"

void Solution::print_undelmiter_routes() {
    for(int i = 0; i < undelmiter_planned_route.size(); ++i) {
        cout << undelmiter_planned_route[i] << " ";
    }
    cout << endl;
}

void Solution::print_delmiter_routes() {
    for(auto route : routes) {
        cout << route<<endl;
    }
}

void Solution::get_random_permutation() {
    vector<int> perm(instance.customerSize);
    for(int i = 0; i < instance.customerSize; ++i) {
        perm[i] = i + 1;
    }
    for(int i = 0; i < instance.customerSize; ++i) {
        int idx = rand() % instance.customerSize;
        swap(perm[i], perm[idx]);
    }
    undelmiter_planned_route = perm;
}

bool Solution::check_feasible() {
    set<int> cus_set(undelmiter_planned_route.begin(),undelmiter_planned_route.end());

    if(undelmiter_planned_route.size() > instance.customerSize) {
        cout << "There are duplicate customers in the solution path" << endl;
        return false;
    }

    for(int i = 1; i <= instance.customerSize; ++i) {
        if(cus_set.find(i) == cus_set.end()) {
            cout << "Solution path missing customers" << endl;
            return false;
        }
    }

    return true;
}

bool Solution::check_time_window(vector<int>& route) {
    double time = 0;

    for(int i = 1; i < route.size(); ++i) {
        int before_id = route[i - 1];
        int cur_id = route[i];
        time = max(time,instance.startTime[before_id])+instance.serviceTime[before_id]+instance.timeMatrix[before_id][cur_id];
        if(time > instance.endTime[cur_id]) {
            return false;
        }
    }
    return true;
}

bool Solution::check_capacity(vector<int>& route) {
    double capacity = 0;
    for(int cus_id : route) {
        capacity += instance.demand[cus_id];
    }
    if(capacity > max_capacity) {
        return false;
    }

    for(int i = 1; i < route.size(); ++i) {
        int current_customer = route[i];
        capacity  = capacity - instance.demand[current_customer] + instance.pickup[current_customer];

        if(capacity > max_capacity) {
            return false;
        }
    }

    return true;

}

bool Solution::is_violate_constraint(vector<int>& route) {
    if(!check_time_window(route)) {
        return true;
    }
    if(!check_capacity(route)) {
        return true;
    }
    return false;
}

double Solution::get_route_dis(vector<int>& route) {
    double dis = 0;
    for(int i = 0; i < route.size() - 1; ++i) {
        dis += instance.distMatrix[route[i]][route[i + 1]];
    }
    return dis;
}

double Solution::get_total_dis() {
    double total_dis = 0;
    for(auto route : routes) {
        total_dis += get_route_dis(route);
    }
    return total_dis;
}

double Solution::get_weight_delta_distance(vector<int> &route, int insert_pos) {
    if(is_violate_constraint(route)) {
        return numeric_limits<double>::infinity();
    }
    int before_cus_id = route[insert_pos - 1];
    int cur_cus_id = route[insert_pos];
    int next_cus_id = route[insert_pos + 1];


    double delta_dis = instance.distMatrix[before_cus_id][cur_cus_id] +
                       instance.distMatrix[cur_cus_id][next_cus_id] -
                       instance.distMatrix[before_cus_id][next_cus_id];
    return delta_dis * dis_factor;
}

void Solution::construct_solution_v2() {
    decode_split(undelmiter_planned_route, instance.customerSize);
}

void Solution::construct_solution() {
    routes.clear();
    vector<int> route1 = {0,undelmiter_planned_route[0],0};
    routes.push_back(route1);
    v_num = 1;

    for(int i = 1; i < instance.customerSize; ++i) {
        int cus_id = undelmiter_planned_route[i];
        if(cus_id == 0) cout <<"-------------------------------------------"<<endl;
        int best_v = -1;
        int best_insert_post = -1;
        double min_delta_tc1 = numeric_limits<double>::infinity();

        for(int r_id = 0; r_id < routes.size(); r_id++) {
            vector<int>& current_route = routes[r_id];
            size_t route_len = current_route.size();

            for(int pos = 1; pos < route_len; pos++) {
                current_route.insert(current_route.begin() + pos,cus_id);
                double delta_dis = get_weight_delta_distance(current_route,pos);
                if(delta_dis < min_delta_tc1) {
                    min_delta_tc1 = delta_dis;
                    best_v = r_id;
                    best_insert_post = pos;
                }
                current_route.erase(current_route.begin()+pos);
            }
        }

        double dis = instance.distMatrix[0][cus_id] + instance.distMatrix[cus_id][0];
        double min_delta_tc2 = vehicle_factor + dis * dis_factor;

        if(min_delta_tc1 <= min_delta_tc2) {
            vector<int>& current_route = routes[best_v];
            current_route.insert(current_route.begin() + best_insert_post,cus_id);
        }else {
            vector<int> new_route = {0, cus_id, 0};
            routes.push_back(new_route);
            v_num++;
        }
    }
}

double Solution::evaluate() {
    distance = get_total_dis();
    v_num = routes.size();
    vehilcle_cost = v_num * vehicle_factor;
    distance_cost = distance * dis_factor;
    total_cost = vehilcle_cost + distance_cost;
    return total_cost;
}
void Solution::deepCopy(Solution* other) {
    undelmiter_planned_route = other->undelmiter_planned_route;
    routes = other->routes;
    total_cost = other->total_cost;
    v_num = other->v_num;
    vehilcle_cost = other->vehilcle_cost;
    distance = other->distance;
    distance_cost = other->distance_cost;
}

Solution::VRPEnv::VRPEnv(Solution &sln)
    : sln_(sln),initSln_(sln), initState_(toTensor(sln_)), state_(initState_),maxNoImprove(maxNoImprove){}

Solution::Agent::Agent(int num_inputs, int num_actions, int hidden_size,
             double epsilon_start, double epsilon_end, int num_steps,
             int update_target, double gamma, int batch_size,
             int memory_capacity, double tau, Solution &sln,string qNetfilePath,string qTargetfilePath)
    : env_(sln),
      q_net_(num_inputs, num_actions, hidden_size),
      q_target_(num_inputs, num_actions, hidden_size),
      replay_memory_(memory_capacity),
      optimizer_(q_net_.parameters(), torch::optim::AdamOptions(1e-3)),
      num_actions_(num_actions),
      epsilon_by_step_([epsilon_start, epsilon_end,
                        num_steps]() -> std::function<double(int)> {
          double epsilon_coefficient =
              log(epsilon_end / epsilon_start) / num_steps;
          return [epsilon_start, epsilon_coefficient](int step) {
              return epsilon_start * exp(epsilon_coefficient * step);
          };
      }()),
      batch_size_(batch_size),
      update_target_(update_target),
      gamma_(gamma),
      tau_(tau),
      num_steps_(num_steps),
      step_count_(0),
      max_no_improve_(0),
      episode_count_(0) {
        std::ifstream file(qNetfilePath);
        if(file.good()){
            q_net_.load_weights(qNetfilePath);
            q_target_.load_weights(qTargetfilePath);
        }else{

        }
    }

torch::Tensor Solution::VRPEnv::reset(){
	//If written like this, from now on, the state_ and sln_ will be out of sync, and each training will be inconsistent, so an init_solution may still be needed.																																							  
    env_step_count_ = 0;
    step_no_improve = 0;
    state_ = toTensor(sln_);
    return state_;
}

std::tuple<torch::Tensor, double, bool> Solution::VRPEnv::step(int action) {
    ++env_step_count_;
    double reward;
    if(action == 0) {
        reward = sln_.best_inter_exchange();
    }
    else if(action == 1) {
        reward = sln_.best_inter_relocate();
    }
    else if(action == 2) {
        reward = sln_.best_2_opt_star();
    }else if(action == 3) {
        reward = sln_.best_or_opt();
    }else if(action == 4) {
        reward = sln_.best_CROSS_exchange();
    }

    if(reward <= 0) ++step_no_improve;
    if(reward>0){//update the state only  if the routes are changed
        sln_.delete_delimiter();
        state_ = toTensor(sln_);
         step_no_improve = 0;
    }

    bool done = false;
    if(step_no_improve >= max_no_imp) {
        done = true;
    }
    return std::make_tuple(state_, reward, done);
}


Solution::ReplayMemory::ReplayMemory(int capacity)
    : capacity_(capacity), generator_(std::random_device()()) {}

void Solution::ReplayMemory::push(const torch::Tensor &state, const int action,
                        const double reward, const torch::Tensor &next_state,
                        const bool done) {
    if (memory_.size() == capacity_) {
        memory_.pop_back();
    }
    memory_.push_front({state, action, reward, next_state, done});
}

std::vector<Solution::Experience> Solution::ReplayMemory::sample(int batch_size) {
    std::vector<Experience> batch;
    std::uniform_int_distribution<int> dist(0, memory_.size() - 1);

    for (int i = 0; i < batch_size; ++i) {
        int idx = dist(generator_);
        batch.push_back(memory_[idx]);
    }

    return batch;
}

int Solution::ReplayMemory::size() const { return memory_.size(); }

void Solution::Agent::act(torch::Tensor &state, int &action, double &reward,
                torch::Tensor &next_state, bool &done) {
    if (step_count_ % update_target_ == 0) {
        // Update the target network with the weights of the current Q network
        auto q_params = q_net_.parameters();
        auto target_params = q_target_.parameters();
        for (size_t i = 0; i < q_params.size(); ++i) {
            auto &target_param = target_params[i];
            auto &q_param = q_params[i];
            target_param.data().copy_(tau_ * q_param.data() +
                                      (1.0 - tau_) * target_param.data());
        }

        // Ensure that the parameters of the two networks are the same
        TORCH_CHECK(q_params.size() == target_params.size(),
                    "Number of parameters must match.");
        for (size_t i = 0; i < q_params.size(); ++i) {
            auto &target_param = target_params[i];
            auto &q_param = q_params[i];
            TORCH_CHECK(target_param.data().equal(q_param.data()),
                        "Parameters must match.");
        }
    }

    double epsilon = epsilon_by_step_(step_count_);
    if (torch::rand({1}).item().toDouble() > epsilon) {
        auto q_values = q_net_.forward(state);
        action = q_values.argmax().item().toInt();
    } else {
        action = torch::randint(num_actions_, {1}).item().toInt();
    }

    std::tie(next_state, reward, done) = env_.step(action);
    replay_memory_.push(state, action, reward, next_state, done);
    state = next_state;
    step_count_++;
}

void Solution::Agent::learn() {
    if (replay_memory_.size() < batch_size_) {
        return;
    }
    optimizer_.zero_grad();

    auto batch = replay_memory_.sample(batch_size_);
    std::vector<torch::Tensor> states, next_states;
    std::vector<int> actions;
    std::vector<double> rewards;
    std::vector<bool> dones;
    for (const auto &experience : batch) {
        states.push_back(experience.state);
        actions.push_back(experience.action);
        rewards.push_back(experience.reward);
        next_states.push_back(experience.next_state);
        dones.push_back(experience.done);
    }
    torch::Tensor state_batch = torch::stack(states, 0);
    torch::Tensor action_batch =
        torch::tensor(actions, torch::TensorOptions().dtype(torch::kInt64))
            .unsqueeze(1);
    torch::Tensor reward_batch =
        torch::tensor(rewards, torch::TensorOptions().dtype(torch::kFloat32))
            .unsqueeze(1);
    torch::Tensor next_state_batch = torch::stack(next_states, 0);
    std::vector<int> done_int(dones.begin(), dones.end());
    torch::Tensor done_batch =
        torch::tensor(done_int, torch::TensorOptions().dtype(torch::kInt32))
            .unsqueeze(1);

    auto q_values =
        q_net_.forward(state_batch).gather(1, action_batch).squeeze(1);
    auto next_q_values =
        std::get<0>(q_target_.forward(next_state_batch).max(1));
    auto expected_q_values = reward_batch.view({-1}) + gamma_ * next_q_values;

    auto loss = torch::mse_loss(q_values, expected_q_values);
    loss.backward();
    optimizer_.step();

}

void Solution::Agent::train(int kNumEpisodes,string q_net_path,string q_target_path) {
    torch::Tensor state = env_.reset();
    int action;
    double reward;
    torch::Tensor next_state;
    bool done;

    int num[7] = {0};
    dqn_total_imp = 0;
    while (episode_count_ < kNumEpisodes) {
        if (episode_count_ % 100 == 0 && episode_count_ / 100 > 0) {
            std::cout << "episode_count_/kNumEpisodes = " << episode_count_
                      << "/" << kNumEpisodes << std::endl;
        }
        act(state, action, reward, next_state, done);
        dqn_total_imp += reward;
        num[action]++;

        //training switch
        //learn();


        if (done) {
            state = env_.reset();
            episode_count_++;
        } else {
            state = next_state;
        }
    }

    //save the network weights
    //q_net_.save_weights(q_net_path);
    //q_target_.save_weights(q_target_path);
}

torch::Tensor Solution::VRPEnv::toTensor(Solution &solution){
    vector<double> assign(1000,0);
    for(int i = 0; i < solution.instance.customerSize; ++i) {
        assign[i] = static_cast<double>(solution.undelmiter_planned_route[i]);
    }

    size_t n = assign.size();
    float *data = new float[n];
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(assign[i]);
    }
    torch::Tensor tensor =
        torch::from_blob(data, {1000}, torch::kFloat32);
    return tensor;
}



void Solution::dqnSearch() {
    const int kBatchSize = 4;
    const double kGamma = 0.90;
    const double kEpsilonStart = 1.0;
    const double kEpsilonEnd = 0.01;
    const int kNumSteps = 1000000;
    const int kUpdateTarget = 1000;
    const int kNumEpisodes = 1;
    const int kHiddenSize = 128;
    const int kMemoryCapacity = 1000;
    const double kTau = 1.0; // the target network update rate
    int modeNum = 4;
    string q_net_path = "q_net_.pt";
    string q_target_path = "q_target.pt";
    std::unique_ptr<Agent> agent=std::make_unique<Agent>(1000, 5, kHiddenSize, kEpsilonStart, kEpsilonEnd, kNumSteps,
           kUpdateTarget, kGamma, kBatchSize, kMemoryCapacity, kTau, *this,q_net_path,q_target_path);
    agent->train(kNumEpisodes,q_net_path,q_target_path);
}

double Solution::twoOpt() {
    int try2Opt=0;
    vector<vector<int>> decode_routes;
    int random_s, random_t;
    while(decode_routes.empty()){
        try2Opt+=1;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, undelmiter_planned_route.size()-reverseLen);
        random_s = dis(gen);
        int delta_idx=1;
        while (delta_idx<reverseLen && decode_routes.empty()){
            std::vector<int> temp_undelmiter_route = undelmiter_planned_route;
            random_t = random_s + delta_idx;
            std::reverse(temp_undelmiter_route.begin() + random_s, temp_undelmiter_route.begin() + random_t+1);
            decode_routes = decode_split_2opt(temp_undelmiter_route, ls_ratio,temp_undelmiter_route.size());
            delta_idx +=1;
        }
    }
    cout<< "twoOpt::try2Opt: " << try2Opt <<" ,reward: " << two_opt_imp<<" ,random_s: " << random_s<<" ,random_t: " << random_t<<" ,reward: " << two_opt_imp << endl;
    routes=decode_routes;
    return two_opt_imp;
}

double Solution::two_opt() {
    double best_imp = 0.0;
    int best_v = -1;
    int best_i = -1;
    bool improvement = false;

    for(int i = 0; i < routes.size(); ++i) {
        vector<int> route = routes[i];
        for(int cus = 1; cus < route.size() - 2; ++cus) {
            double old_dis = instance.distMatrix[route[cus-1]][route[cus]] + instance.distMatrix[route[cus]][route[cus+1]] + instance.distMatrix[route[cus+1]][route[cus+2]];
            double new_dis = instance.distMatrix[route[cus-1]][route[cus+1]] + instance.distMatrix[route[cus+1]][route[cus]] + instance.distMatrix[route[cus]][route[cus+2]];
            if(new_dis - old_dis < best_imp) {
                vector<int>temp_route = route;
                swap(temp_route[i],temp_route[i+1]);
                if(!is_violate_constraint(temp_route)) {
                    best_imp = new_dis - old_dis;
                    best_v = i;
                    best_i = cus;
                    improvement = true;
                }
            }
        }

    }

    if(improvement) {
        vector<int> &route = routes[best_v];
        swap(route[best_i],route[best_i+1]);
    }
    double reward = (-1) * best_imp *  dis_factor;
    return reward;
}

double Solution::best_2_opt_star() {
    int best_route1 = -1, best_route2 = -1;
    int best_i = -1, best_j = -1;
    double sum_imp = 0;

    for(int i = 0; i < routes.size(); ++i) {
        if(routes[i].empty()) continue;;
        for(int j = i + 1; j < routes.size(); ++j) {
            if(i == j || routes[j].empty()) continue;
            int len1 = routes[i].size();
            int len2 = routes[j].size();
            bool improvement = false;
            double best_imp = 0;
            for(int cus1 = 1; cus1 < len1 - 1; ++cus1) {
                for(int cus2 = 1; cus2 < len2 - 1; ++cus2) {

                    vector<int> route1(routes[i].begin(),routes[i].begin()+cus1+1);
                    vector<int> route2(routes[j].begin(),routes[j].begin()+cus2+1);
                    route1.insert(route1.end(),routes[j].begin()+cus2+1,routes[j].end());
                    route2.insert(route2.end(),routes[i].begin()+cus1+1,routes[i].end());
                    if(is_violate_constraint(route1)||is_violate_constraint(route2)) {
                        continue;
                    }

                    double delta_cost = instance.distMatrix[routes[i][cus1]][routes[j][cus2 + 1]] -
                                        instance.distMatrix[routes[i][cus1]][routes[i][cus1 + 1]] +
                                        instance.distMatrix[routes[j][cus2]][routes[i][cus1 + 1]] -
                                        instance.distMatrix[routes[j][cus2]][routes[j][cus2 + 1]];
                    if(delta_cost < best_imp) {
                        best_imp = delta_cost;
                        best_route1 = i;
                        best_route2 = j;
                        best_i = cus1;
                        best_j = cus2;
                        improvement = true;
                    }
                }
            }
            if(improvement) {
                vector<int> route1(routes[best_route1].begin(),routes[best_route1].begin()+best_i+1);
                vector<int> route2(routes[best_route2].begin(),routes[best_route2].begin()+best_j+1);
                route1.insert(route1.end(),routes[best_route2].begin()+best_j+1,routes[best_route2].end());
                route2.insert(route2.end(),routes[best_route1].begin()+best_i+1,routes[best_route1].end());
                routes[best_route1] = route1;
                routes[best_route2] = route2;
                sum_imp+=best_imp;
            }
        }
    }

    double reward = (-1) * sum_imp * dis_factor; 
    return reward;
}

double Solution::best_or_opt() {
    int best_v1 = -1, best_v2 = -1;
    int best_i = -1, best_j = -1;
    double sum_imp = 0;

    for(int i = 0; i < routes.size(); ++i) {
        if(routes[i].empty()) continue;
        for(int j = 0; j < routes.size(); ++j) {
            if(i == j || routes[j].empty()) continue;
            int len1 = routes[i].size();
            int len2 = routes[j].size();
            bool improvement = false;
            double best_imp = 0;

            for(int cus1 = 1; cus1 < len1 - 2; ++cus1) {
                vector<int> route1 = routes[i];
                int temp_cust1 = route1[cus1];
                int temp_cust2 = route1[cus1 + 1];
                route1.erase(route1.begin()+cus1);
                route1.erase(route1.begin()+cus1);

                for(int cus2 = 0; cus2 < len2 - 1; ++cus2) {
                    vector<int> route2 = routes[j];
                    route2.insert(route2.begin()+cus2+1,temp_cust1);
                    route2.insert(route2.begin()+cus2+2,temp_cust2);
                    if(is_violate_constraint(route1) || is_violate_constraint(route2)) {
                        continue;
                    }
                    double delta_cost1 = instance.distMatrix[routes[i][cus1 - 1]][routes[i][cus1 + 2]] -
                                         instance.distMatrix[routes[i][cus1 - 1]][routes[i][cus1]] -
                                         instance.distMatrix[routes[i][cus1+1]][routes[i][cus1+2]];
                    double delta_cost2 = instance.distMatrix[routes[j][cus2]][routes[i][cus1]] +
                                         instance.distMatrix[routes[i][cus1+1]][routes[j][cus2+1]] -
                                         instance.distMatrix[routes[j][cus2]][routes[j][cus2+1]];
                    if(delta_cost1 + delta_cost2 < best_imp) {
                        best_imp = delta_cost1 + delta_cost2;
                        best_v1 = i;
                        best_v2 = j;
                        best_i = cus1;
                        best_j = cus2;
                        improvement = true;
                    }
                }
            }
            if(improvement) {
                int cus1 = routes[best_v1][best_i];
                int cus2 = routes[best_v1][best_i + 1];

                routes[best_v1].erase(routes[best_v1].begin() + best_i);
                routes[best_v1].erase(routes[best_v1].begin() + best_i );
                if (routes[best_v1].size() == 2) {
                    routes.erase(routes.begin() + best_v1);
                    if (best_v1 < best_v2) {
                        j=-1;
                        best_v2 -= 1;
                    }
                }
                routes[best_v2].insert(routes[best_v2].begin() + best_j + 1, cus1);
                routes[best_v2].insert(routes[best_v2].begin() + best_j + 2, cus2);
                sum_imp+=best_imp;
                if(i>=routes.size()) {
                    break;
                }
            }
        }
    }

    double reward = (-1) * sum_imp * dis_factor; 
    return reward;
}

double Solution::best_CROSS_exchange() {
    int best_v1 = -1, best_v2 = -1;
    int best_i = -1, best_j = -1;
    double sum_imp = 0;

    for(int i = 0; i < routes.size(); ++i) {
        if(routes[i].empty()) continue;
        for(int j = i + 1; j < routes.size(); ++j) {
            if(i == j || routes[j].empty()) continue;
            int len1 = routes[i].size();
            int len2 = routes[j].size();
            bool improvement = false;
            double best_imp = 0;

            for(int cus1 = 1; cus1 < len1 - 2; ++cus1) {
                for(int cus2 = 1; cus2 < len2 - 2; ++cus2) {
                    vector<int> route1 = routes[i];
                    vector<int> route2 = routes[j];

                    int temp_cust1 = route1[cus1];
                    int temp_cust2 = route1[cus1 + 1];

                    int temp_cust3 = route2[cus2];
                    int temp_cust4 = route2[cus2 + 1];

                    route1.erase(route1.begin()+cus1);
                    route1.erase(route1.begin()+cus1);

                    route2.erase(route2.begin()+cus2);
                    route2.erase(route2.begin()+cus2);

                    route1.insert(route1.begin()+cus1,temp_cust3);
                    route1.insert(route1.begin()+cus1+1,temp_cust4);

                    route2.insert(route2.begin()+cus2,temp_cust1);
                    route2.insert(route2.begin()+cus2+1,temp_cust2);

                    if(is_violate_constraint(route1) || is_violate_constraint(route2)) {
                        continue;
                    }

                    double delta_cost1 = instance.distMatrix[routes[i][cus1 - 1]][routes[j][cus2]] +
                                         instance.distMatrix[routes[j][cus2+1]][routes[i][cus1+2]] -
                                         instance.distMatrix[routes[i][cus1-1]][routes[i][cus1]] -
                                         instance.distMatrix[routes[i][cus1+1]][routes[i][cus1+2]];
                    double delta_cost2 = instance.distMatrix[routes[j][cus2 - 1]][routes[i][cus1]] +
                                         instance.distMatrix[routes[i][cus1+1]][routes[j][cus2+2]] -
                                         instance.distMatrix[routes[j][cus2-1]][routes[j][cus2]] -
                                         instance.distMatrix[routes[j][cus2+1]][routes[j][cus2+2]];
                    if(delta_cost1 + delta_cost2 < best_imp) {
                        best_imp = delta_cost1 + delta_cost2;
                        best_v1 = i;
                        best_v2 = j;
                        best_i = cus1;
                        best_j = cus2;
                        improvement = true;
                    }
                }
            }
            if(improvement) {
                sum_imp+=best_imp;
                int cus1 = routes[best_v1][best_i];
                int cus2 = routes[best_v1][best_i + 1];

                int cus3 = routes[best_v2][best_j];
                int cus4 = routes[best_v2][best_j + 1];

                routes[best_v1].erase(routes[best_v1].begin() + best_i);
                routes[best_v1].erase(routes[best_v1].begin() + best_i);

                routes[best_v2].erase(routes[best_v2].begin() + best_j);
                routes[best_v2].erase(routes[best_v2].begin() + best_j);

                routes[best_v1].insert(routes[best_v1].begin() + best_i, cus3);
                routes[best_v1].insert(routes[best_v1].begin() + best_i + 1, cus4);

                routes[best_v2].insert(routes[best_v2].begin() + best_j, cus1);
                routes[best_v2].insert(routes[best_v2].begin() + best_j +  1, cus2);
            }
        }
    }

    double reward = (-1) * sum_imp * dis_factor; //The bigger the reward, the better
    return reward;
}

double Solution::best_inter_exchange() {
    int best_v1 = -1, best_v2 = -1, pos1 = -1, pos2 = -1;
    double imp_cost = 0.0;
    double best_imp = 0;
    bool is_improve = false;

    for(int i = 0; i < routes.size(); ++i) {
        if(routes[i].empty()) continue;
        for(int j = i + 1; j < routes.size(); ++j) {
            if(i==j || routes[j].empty()) continue;
            int len1 = routes[i].size();
            int len2 = routes[j].size();
            for(int cus1 = 1; cus1 < len1 -1; ++cus1) {
                for(int cus2 = 1; cus2 < len2 - 1; ++cus2) {
                    vector<int> route1 = routes[i];
                    vector<int> route2 = routes[j];

                    int temp_cus1 = route1[cus1];
                    route1.erase(route1.begin()+cus1);
                    int temp_cus2 = route2[cus2];
                    route2.erase(route2.begin()+cus2);

                    route1.insert(route1.begin()+cus1,temp_cus2);
                    route2.insert(route2.begin()+cus2,temp_cus1);

                    if(is_violate_constraint(route1)||is_violate_constraint(route2)) {
                        continue;
                    }

                    double delta_cost1 = instance.distMatrix[route1[cus1 - 1]][temp_cus2] +
                                         instance.distMatrix[temp_cus2][route1[cus1 + 1]] -
                                         instance.distMatrix[route1[cus1 - 1]][temp_cus1] -
                                         instance.distMatrix[temp_cus1][route1[cus1 + 1]];

                    double delta_cost2 = instance.distMatrix[route2[cus2 - 1]][temp_cus1] +
                                         instance.distMatrix[temp_cus1][route2[cus2 + 1]] -
                                         instance.distMatrix[route2[cus2 - 1]][temp_cus2] -
                                         instance.distMatrix[temp_cus2][route2[cus2 + 1]];

                    if (delta_cost1 + delta_cost2 < 0) {
                        best_imp += delta_cost1 + delta_cost2;
                        best_v1 = i;
                        best_v2 = j;
                        pos1 = cus1;
                        pos2 = cus2;
                        is_improve = true;
                        int cust1 = routes[best_v1][pos1];
                        int cust2 = routes[best_v2][pos2];

                        routes[best_v1].erase(routes[best_v1].begin() + pos1);
                        routes[best_v1].insert(routes[best_v1].begin() + pos1, cust2);

                        routes[best_v2].erase(routes[best_v2].begin() + pos2);
                        routes[best_v2].insert(routes[best_v2].begin() + pos2, cust1);
                    }

                }
            }

        }
    }
    double reward =  (-1) * best_imp * dis_factor; //The bigger the reward, the better
    return reward;
}

double Solution::intra_exchange() {
    double imp_cost = 0.0;
    bool is_improve = false;

    for (int route_idx = 0; route_idx < routes.size(); ++route_idx) {
        std::vector<int>& route = routes[route_idx];
        for (int i = 1; i < route.size() - 1; ++i) {
            for (int j = 1; j < route.size() - 1; ++j) {
                if (i == j) {
                    continue;
                }
                int i_idx = std::min(i, j);
                int j_idx = std::max(i, j);

                double old_dist, new_dist;
                if (j_idx == i_idx + 1) {
                    old_dist = instance.distMatrix[route[i_idx - 1]][route[i_idx]] +
                               instance.distMatrix[route[i_idx]][route[j_idx]] +
                               instance.distMatrix[route[j_idx]][route[j_idx + 1]];
                    new_dist = instance.distMatrix[route[i_idx - 1]][route[j_idx]] +
                               instance.distMatrix[route[j_idx]][route[i_idx]] +
                               instance.distMatrix[route[i_idx]][route[j_idx + 1]];
                } else {
                    old_dist = instance.distMatrix[route[i_idx - 1]][route[i_idx]] +
                               instance.distMatrix[route[i_idx]][route[i_idx + 1]] +
                               instance.distMatrix[route[j_idx - 1]][route[j_idx]] +
                               instance.distMatrix[route[j_idx]][route[j_idx + 1]];
                    new_dist = instance.distMatrix[route[i_idx - 1]][route[j_idx]] +
                               instance.distMatrix[route[j_idx]][route[i_idx + 1]] +
                               instance.distMatrix[route[j_idx - 1]][route[i_idx]] +
                               instance.distMatrix[route[i_idx]][route[j_idx + 1]];
                }

                if (new_dist < old_dist) {
                    std::vector<int> temp_route = route;
                    std::swap(temp_route[i_idx], temp_route[j_idx]);

                    if (!is_violate_constraint(temp_route)) {
                        is_improve = true;
                        imp_cost += old_dist - new_dist;
                        route = temp_route;
                    }
                }
            }
        }
    }

    double reward = (-1) * imp_cost * dis_factor;  
    return reward;
}

double Solution::best_intra_exchange() {
    double imp_cost = 0.0;
    int best_pos1 = -1;
    int best_pos2 = -1;
    int best_v = -1;
    bool is_imp = false;

    for (int vid = 0; vid < routes.size(); ++vid) {
        std::vector<int>& route = routes[vid];
        for (int i = 1; i < route.size() - 1; ++i) {
            for (int j = i + 1; j < route.size() - 1; ++j) {
                double old_dist, new_dist;

                if (j == i + 1) {  
                    old_dist = instance.distMatrix[route[i - 1]][route[i]] + instance.distMatrix[route[i]][route[j]] +
                               instance.distMatrix[route[j]][route[j + 1]];
                    new_dist = instance.distMatrix[route[i - 1]][route[j]] + instance.distMatrix[route[j]][route[i]] +
                               instance.distMatrix[route[i]][route[j + 1]];
                } else {
                    old_dist = instance.distMatrix[route[i - 1]][route[i]] + instance.distMatrix[route[i]][route[i + 1]] +
                               instance.distMatrix[route[j - 1]][route[j]] + instance.distMatrix[route[j]][route[j + 1]];
                    new_dist = instance.distMatrix[route[i - 1]][route[j]] + instance.distMatrix[route[j]][route[i + 1]] +
                               instance.distMatrix[route[j - 1]][route[i]] + instance.distMatrix[route[i]][route[j + 1]];
                }

                if (new_dist - old_dist < imp_cost) {
                    std::vector<int> temp_route = route;
                    std::swap(temp_route[i], temp_route[j]);

                    if (!is_violate_constraint(temp_route)) {
                        is_imp = true;
                        imp_cost = new_dist - old_dist;
                        best_v = vid;
                        best_pos1 = i;
                        best_pos2 = j;
                    }
                }
            }
        }
    }

    if (is_imp) {
        std::vector<int>& route = routes[best_v];
        std::swap(route[best_pos1], route[best_pos2]);
    }

    double reward = (-1) * imp_cost * dis_factor;
    return reward;
}

double Solution::best_inter_relocate() {
    double best_imp_cost = 0.0;
    int best_from_v = -1;
    int best_cus_idx = -1;
    int best_relocate_v = -1;
    int best_relocate_pos = -1;

    for (int vehicle_id = 0; vehicle_id < routes.size(); ++vehicle_id) {
        if (vehicle_id >= routes.size() || routes[vehicle_id].empty()) {
            continue;
        }
        for (int cus_idx = 1; cus_idx < routes[vehicle_id].size() - 1; ++cus_idx) {
            int relocate_v, relocate_pos;
            double imp;
            std::tie(relocate_v, relocate_pos, imp) = relocate(vehicle_id, cus_idx, routes);

            if (imp < 0) {
                best_imp_cost += imp;
                best_from_v = vehicle_id;
                best_cus_idx = cus_idx;
                best_relocate_v = relocate_v;
                best_relocate_pos = relocate_pos;

                int cus = routes[best_from_v][best_cus_idx];
                routes[best_from_v].erase(routes[best_from_v].begin() + best_cus_idx);
                if (routes[best_from_v].size() == 2) {
                    routes.erase(routes.begin() + best_from_v);
                    if (best_from_v < best_relocate_v) {
                        best_relocate_v -= 1;
                        cus_idx = 0;
                    }
                }
                if (best_relocate_v == -1 || best_relocate_pos == -1) {
                    std::vector<int> new_route = {0, cus, 0};
                    routes.push_back(new_route);
                } else {
                    routes[best_relocate_v].insert(routes[best_relocate_v].begin() + best_relocate_pos, cus);
                }
                if(vehicle_id>=routes.size()) {
                    break;
                }
            }
        }
    }


    double reward = (-1) * best_imp_cost;
    return reward;
}

double Solution::best_intra_relocate() {
        double best_imp_cost = 0.0;
        int best_v = -1;
        int best_orign_pos = -1;
        int best_relocate_cus = -1;
        int best_relocate_pos = -1;

        for (int i = 0; i < routes.size(); ++i) {
            int route_length = routes[i].size();
            if (route_length > 3) {
                for (int j = 1; j < route_length - 1; ++j) {
                    std::vector<int> temp_route = routes[i];
                    int customer = routes[i][j];
                    int orign_before_cus = routes[i][j - 1];
                    int orign_after_cus = routes[i][j + 1];
                    double delta1 = instance.distMatrix[orign_before_cus][orign_after_cus] -
                                    instance.distMatrix[orign_before_cus][customer] - instance.distMatrix[customer][orign_after_cus];

                    temp_route.erase(temp_route.begin() + j);
                    double min_delta2 = std::numeric_limits<double>::infinity();
                    int min_idx = -1;

                    for (int k = 1; k < temp_route.size() - 1; ++k) {
                        int before_cus = temp_route[k - 1];
                        int after_cus = temp_route[k];
                        double delta2 = instance.distMatrix[before_cus][customer] + instance.distMatrix[customer][after_cus] -
                                        instance.distMatrix[before_cus][after_cus];

                        temp_route.insert(temp_route.begin() + k, customer);
                        if (is_violate_constraint(temp_route)) {
                            temp_route.erase(temp_route.begin() + k);
                            continue;
                        }
                        if (delta2 < min_delta2) {
                            min_delta2 = delta2;
                            min_idx = k;
                        }
                        temp_route.erase(temp_route.begin() + k);
                    }

                    if (min_delta2 + delta1 < best_imp_cost) {
                        best_imp_cost = min_delta2 + delta1;
                        best_v = i;
                        best_orign_pos = j;
                        best_relocate_cus = customer;
                        best_relocate_pos = min_idx;
                    }
                }
            }
        }

        if (best_imp_cost < 0.0) {
            routes[best_v].erase(routes[best_v].begin() + best_orign_pos);
            routes[best_v].insert(routes[best_v].begin() + best_relocate_pos, best_relocate_cus);
        }

        double reward = (-1) * best_imp_cost * dis_factor;
        return reward;
    }

tuple<int, int, double> Solution::relocate(int v_id, int cus_idx, vector<vector<int>> plan_routes) {
    int cus = plan_routes[v_id][cus_idx];
    int best_relocate_v = -1;
    int best_relocate_pos = -1;
    double best_relocate_delta2 = std::numeric_limits<double>::infinity();
    double best_relocate_delta1 = std::numeric_limits<double>::infinity();
    bool is_improve = false;
    double imp_cost = 0.0;
    bool is_delete_route = false;
    bool is_new_route = false;

    for (int j = 0; j < plan_routes.size(); ++j) {
        if (j == v_id || plan_routes[j].empty()) {
            continue;
        }
        int len2 = plan_routes[j].size();
        for (int pos = 1; pos < len2; ++pos) {
            std::vector<int> route2 = plan_routes[j];
            route2.insert(route2.begin() + pos, cus);
            if (is_violate_constraint(route2)) {
                continue;
            }
            double delta_cost2 = (instance.distMatrix[route2[pos - 1]][cus] + instance.distMatrix[cus][route2[pos + 1]]
                                - instance.distMatrix[route2[pos - 1]][route2[pos + 1]]) * dis_factor;
            if (delta_cost2 < best_relocate_delta2) {
                best_relocate_delta2 = delta_cost2;
                best_relocate_v = j;
                best_relocate_pos = pos;
            }
        }
    }

    double new_route_cost = (instance.distMatrix[0][cus] + instance.distMatrix[cus][0]) * dis_factor + vehicle_factor;

    best_relocate_delta1 = 0.0;
    if (plan_routes[v_id].size() == 3) {
        best_relocate_delta1 = -(instance.distMatrix[0][cus] + instance.distMatrix[cus][0]) * dis_factor - vehicle_factor;
    } else {
        int before_cus = plan_routes[v_id][cus_idx - 1];
        int next_cus = plan_routes[v_id][cus_idx + 1];
        best_relocate_delta1 = (instance.distMatrix[before_cus][next_cus] - instance.distMatrix[before_cus][cus]
                                - instance.distMatrix[cus][next_cus]) * dis_factor;
    }

    if (best_relocate_delta2 <= new_route_cost && best_relocate_delta1 + best_relocate_delta2 < 0.0) {
        imp_cost = best_relocate_delta1 + best_relocate_delta2;
    } else if (new_route_cost + best_relocate_delta1 < 0.0) {
        is_improve = true;
        imp_cost = new_route_cost + best_relocate_delta1;
        best_relocate_v = -1;
        best_relocate_pos = -1;
    }

    return std::make_tuple(best_relocate_v, best_relocate_pos, imp_cost);
}

void Solution::delete_delimiter() {
    undelmiter_planned_route.clear();
    int cnt = 0;
    for(auto route: routes) {
        for(int idx = 1; idx < route.size() - 1; ++idx) {
            undelmiter_planned_route.push_back(route[idx]);
            cnt++;
        }
    }

    assert(cnt == instance.customerSize);
}

void Solution::mutation() {
    std::vector<int> temp_undelmiter_route = undelmiter_planned_route;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_perturb_action(0, 2);
    int perturb_action = dis_perturb_action(gen);
    int random_s,random_t;
    if(perturb_action == 0){
        std::uniform_int_distribution<> dis(0, temp_undelmiter_route.size() - 2);
        random_s = dis(gen);
        std::uniform_int_distribution<> dis2(1, 6);
        random_t = random_s + dis2(gen);
        while (random_t >= temp_undelmiter_route.size()){
            random_t =  random_s + dis2(gen);
        }
        std::reverse(temp_undelmiter_route.begin() + random_s, temp_undelmiter_route.begin() + random_t+1);
    } else if (perturb_action == 1){
        std::uniform_int_distribution<> dis_pos(0, temp_undelmiter_route.size() - 1);
        random_s = dis_pos(gen);
        random_t = dis_pos(gen);
        while (random_t == random_s){
            random_t =  dis_pos(gen);
        }
        std::swap(temp_undelmiter_route[random_s], temp_undelmiter_route[random_t]);
    }else{
        std::uniform_int_distribution<> dis_pos(0, temp_undelmiter_route.size() - 1);
        random_s = dis_pos(gen);
        random_t = dis_pos(gen);
        while (random_t == random_s){
            random_t =  dis_pos(gen);
        }
        int element = temp_undelmiter_route[random_s];
        temp_undelmiter_route.erase(temp_undelmiter_route.begin() + random_s);
        if (random_s < random_t) {
            --random_t;
        }
        temp_undelmiter_route.insert(temp_undelmiter_route.begin() + random_t, element);
    }
    undelmiter_planned_route=temp_undelmiter_route;
    construct_solution_v2();
}

vector<vector<int>> Solution::perturbation(double perturb_ratio) {
    std::vector<int> temp_undelmiter_route = undelmiter_planned_route;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_perturb_action(0, 2);
    int perturb_action = dis_perturb_action(gen);
    int random_s,random_t;
    if(perturb_action == 0){
        std::uniform_int_distribution<> dis(0, temp_undelmiter_route.size() - 2);
        random_s = dis(gen);
        std::uniform_int_distribution<> dis2(1, 6);
        random_t = random_s + dis2(gen);
        while (random_t >= temp_undelmiter_route.size()){
            random_t =  random_s + dis2(gen);
        }
        std::reverse(temp_undelmiter_route.begin() + random_s, temp_undelmiter_route.begin() + random_t+1);
    } else if (perturb_action == 1){
        std::uniform_int_distribution<> dis_pos(0, temp_undelmiter_route.size() - 1);
        random_s = dis_pos(gen);
        random_t = dis_pos(gen);
        while (random_t == random_s){
            random_t =  dis_pos(gen);
        }
        std::swap(temp_undelmiter_route[random_s], temp_undelmiter_route[random_t]);
    }else{
        std::uniform_int_distribution<> dis_pos(0, temp_undelmiter_route.size() - 1);
        random_s = dis_pos(gen);
        random_t = dis_pos(gen);
        while (random_t == random_s){
            random_t =  dis_pos(gen);
        }
        int element = temp_undelmiter_route[random_s];
        temp_undelmiter_route.erase(temp_undelmiter_route.begin() + random_s);
        if (random_s < random_t) {
            --random_t;
        }
        temp_undelmiter_route.insert(temp_undelmiter_route.begin() + random_t, element);
    }
    vector<vector<int>> decode_routes = decode_split_perturb(temp_undelmiter_route, perturb_ratio, temp_undelmiter_route.size()*perturb_max_k_ratio);
    if(!decode_routes.empty()){
        cout <<"perturb_action: "<< perturb_action<< " ,random_s: "<<random_s<<", random_t:"<<random_t<<endl;
    }
    return decode_routes;
}

std::vector<std::vector<int>> Solution::decode_split_2opt(std::vector<int>& temp_undelmiter_route, double imp_rate, int max_k_size) {
    size_t count = temp_undelmiter_route.size() + 1;
    std::vector<int> sequence = {0};
    sequence.insert(sequence.end(), temp_undelmiter_route.begin(), temp_undelmiter_route.end());
    int n = count;  
    std::vector<double> dis(n, 0);

    for (int i = 1; i < count; ++i) {
        dis[i] = dis[i - 1] + instance.distMatrix[sequence[i - 1]][sequence[i]];
    }

    std::vector<std::vector<double>> v(count, std::vector<double>(max_k_size + 1, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> p(count, std::vector<int>(max_k_size + 1, -1));

    for (int i = 0; i <= max_k_size; ++i) {
        v[0][i] = p[0][i] = 0;  
    }
    for (int k = 1; k <= max_k_size; ++k) {
        for (int j = 1; j < count; ++j) {
            for (int i = j; i > 0; --i) {
                std::vector<int> temp_route = {0};
                temp_route.insert(temp_route.end(), sequence.begin() + i, sequence.begin() + j + 1);
                temp_route.push_back(0);
                if(!check_capacity(temp_route)) {
                    break;
                }
                if(!check_time_window(temp_route)) { 
                    continue;
                }
                if (v[i - 1][k - 1] >= std::numeric_limits<double>::infinity()) {
                    continue;
                }

                double change_tc = (instance.distMatrix[0][sequence[i]] + dis[j] - dis[i] +
                                    instance.distMatrix[sequence[j]][0]) * dis_factor + vehicle_factor;
                if (change_tc + v[i - 1][k - 1] < v[j][k]) {
                    v[j][k] = change_tc + v[i - 1][k - 1];
                    p[j][k] = i - 1;
                }
            }
        }
    }
    double p_min = std::numeric_limits<double>::infinity();
    int k_best = -1;
    for (int i = 1; i <= max_k_size; ++i) {
        if (p_min > v[count - 1][i]) {
            p_min = v[count - 1][i];
            k_best = i;
        }
    }
    std::vector<std::vector<int>> pd_routes;
    if( k_best == -1 || dqn_total_imp + v[count - 1][k_best] >= total_cost*imp_rate){
        return pd_routes;
    }
    two_opt_imp=total_cost-dqn_total_imp-v[count - 1][k_best];
    int current_position = p[count - 1][k_best];
    int end_pos = count;
    while (current_position) {
        if (current_position < 0 || current_position >= n) {
            std::cerr << "cannot split\n";
            exit(EXIT_FAILURE);
        }
        std::vector<int> route;
        route.push_back(0);
        for (int i = current_position + 1; i < end_pos; ++i) {
            route.push_back(sequence[i]);
        }
        route.push_back(0);

        pd_routes.insert(pd_routes.begin(), route);
        end_pos = current_position + 1;
        current_position = p[current_position][k_best - 1];
        k_best -= 1;
    }
    std::vector<int> route;
    route.push_back(0);
    for (int i = current_position + 1; i < end_pos; ++i) {
        route.push_back(sequence[i]);
    }
    route.push_back(0);
    pd_routes.insert(pd_routes.begin(), route);
    return pd_routes;
}

void Solution::decode_split(std::vector<int>& temp_undelmiter_route, int max_k_size) {
    size_t count = temp_undelmiter_route.size() + 1;
    std::vector<int> sequence = {0};
    sequence.insert(sequence.end(), temp_undelmiter_route.begin(), temp_undelmiter_route.end());
    int n = count;  
    std::vector<double> dis(n, 0);

    for (int i = 1; i < count; ++i) {
        dis[i] = dis[i - 1] + instance.distMatrix[sequence[i - 1]][sequence[i]];
    }

    std::vector<std::vector<double>> v(count, std::vector<double>(max_k_size + 1, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> p(count, std::vector<int>(max_k_size + 1, -1));

    for (int i = 0; i <= max_k_size; ++i) {
        v[0][i] = p[0][i] = 0;  
    }
    bool reach_check_point=false;
    double p_min = std::numeric_limits<double>::infinity();
    int k_best = -1;
    bool pruning =true;
    int begin_j=1;
    for (int k = 1; k <= max_k_size; ++k) {
        for (int j = begin_j; j < count; ++j) {
            pruning =true;
            for (int i = j; i > 0; --i) {
                std::vector<int> temp_route = {0};
                temp_route.insert(temp_route.end(), sequence.begin() + i, sequence.begin() + j + 1);
                temp_route.push_back(0);
                if(!check_capacity(temp_route)) {
                    break;
                }
                if(!check_time_window(temp_route)) { 
                    continue;
                }
                if (v[i - 1][k - 1] >= std::numeric_limits<double>::infinity()) {
                    continue;
                }
                pruning =false;
                double change_tc = (instance.distMatrix[0][sequence[i]] + dis[j] - dis[i] +
                                    instance.distMatrix[sequence[j]][0]) * dis_factor + vehicle_factor;
                if (change_tc + v[i - 1][k - 1] < v[j][k]) {
                    v[j][k] = change_tc + v[i - 1][k - 1];
                    p[j][k] = i - 1;
                    begin_j = j+1;
                    if(j==count-1){
                        reach_check_point=true;
                        break;
                    }
                }
            }
            if(pruning) break;
        }
        if(reach_check_point){
            p_min = v[count - 1][k];
            k_best = k;
            break;
        }
    }
    v_num=k_best;
    std::vector<std::vector<int>> pd_routes;
    int current_position = p[count - 1][k_best];
    int end_pos = count;
    while (current_position) {
        if (current_position < 0 || current_position >= n) {
            std::cerr << "cannot split\n";
            exit(EXIT_FAILURE);
        }
        std::vector<int> route;
        route.push_back(0);
        for (int i = current_position + 1; i < end_pos; ++i) {
            route.push_back(sequence[i]);
        }
        route.push_back(0);

        pd_routes.insert(pd_routes.begin(), route);
        end_pos = current_position + 1;
        current_position = p[current_position][k_best - 1];
        k_best -= 1;
    }
    std::vector<int> route;
    route.push_back(0);
    for (int i = current_position + 1; i < end_pos; ++i) {
        route.push_back(sequence[i]);
    }
    route.push_back(0);
    pd_routes.insert(pd_routes.begin(), route);

    routes = std::move(pd_routes); 
    total_cost=p_min;
    vehilcle_cost = v_num * vehicle_factor;
    distance_cost = total_cost - vehilcle_cost;
    distance= round(distance_cost/dis_factor);
}

std::vector<std::vector<int>> Solution::decode_split_perturb(std::vector<int>& temp_undelmiter_route, double perturb_ratio, int max_k_size) {
    size_t count = temp_undelmiter_route.size() + 1;
    std::vector<int> sequence = {0};
    sequence.insert(sequence.end(), temp_undelmiter_route.begin(), temp_undelmiter_route.end());
    int n = count;  
    std::vector<double> dis(n, 0);

    for (int i = 1; i < count; ++i) {
        dis[i] = dis[i - 1] + instance.distMatrix[sequence[i - 1]][sequence[i]];
    }

    std::vector<std::vector<double>> v(count, std::vector<double>(max_k_size + 1, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> p(count, std::vector<int>(max_k_size + 1, -1));

    for (int i = 0; i <= max_k_size; ++i) {
        v[0][i] = p[0][i] = 0;  
    }
    bool reach_check_point=false;
    double p_min = std::numeric_limits<double>::infinity();
    int k_best = -1;
    for (int k = 1; k <= max_k_size; ++k) {
        for (int j = 1; j < count; ++j) {
            for (int i = j; i > 0; --i) {
                std::vector<int> temp_route = {0};
                temp_route.insert(temp_route.end(), sequence.begin() + i, sequence.begin() + j + 1);
                temp_route.push_back(0);
                if(!check_capacity(temp_route)) {
                    break;
                }
                if(!check_time_window(temp_route)) { 
                    continue;
                }
                if (v[i - 1][k - 1] >= std::numeric_limits<double>::infinity()) {
                    continue;
                }

                double change_tc = (instance.distMatrix[0][sequence[i]] + dis[j] - dis[i] +
                        instance.distMatrix[sequence[j]][0]) * dis_factor + vehicle_factor;
                if (change_tc + v[i - 1][k - 1] < v[j][k]) {
                    v[j][k] = change_tc + v[i - 1][k - 1];
                    p[j][k] = i - 1;
                    if(j==count-1){
                        reach_check_point=true;
                    }
                }
            }
        }
        if(reach_check_point){
            if(v[count - 1][k]>=total_cost*perturb_ratio){
                std::vector<std::vector<int>> sol_routes;
                return sol_routes;
            }else{
                p_min = v[count - 1][k];
                k_best = k;
                break;
            }
        }
    }
    std::vector<std::vector<int>> pd_routes;
    int current_position = p[count - 1][k_best];
    int end_pos = count;
    while (current_position) {
        if (current_position < 0 || current_position >= n) {
            std::cerr << "cannot split\n";
            exit(EXIT_FAILURE);
        }
        std::vector<int> route;
        route.push_back(0);
        for (int i = current_position + 1; i < end_pos; ++i) {
            route.push_back(sequence[i]);
        }
        route.push_back(0);

        pd_routes.insert(pd_routes.begin(), route);
        end_pos = current_position + 1;
        current_position = p[current_position][k_best - 1];
        k_best -= 1;
    }
    std::vector<int> route;
    route.push_back(0);
    for (int i = current_position + 1; i < end_pos; ++i) {
        route.push_back(sequence[i]);
    }
    route.push_back(0);
    pd_routes.insert(pd_routes.begin(), route);
    return pd_routes;
}

void Solution::twoOptSwap(std::vector<int>& tour, int i, int k) {
    std::vector<int> newTour;
    for (int c = 0; c <= i - 1; ++c) {
        newTour.push_back(tour[c]);
    }

    for (int c = k; c >= i; --c) {
        newTour.push_back(tour[c]);
    }

    for (int c = k + 1; c < tour.size(); ++c) {
        newTour.push_back(tour[c]);
    }

    tour = newTour;
}

void Solution::twoOptPerturbationIter() {
    double orignal_cost = total_cost;

    bool improvement = true;
    for (size_t i = 0; i < undelmiter_planned_route.size() - 1; i++) {
        for (size_t k = i + 1; k < undelmiter_planned_route.size(); k++) {
            vector<int>& tour = undelmiter_planned_route;
            twoOptSwap(tour, i, k);
        }
    }
}




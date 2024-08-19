#include "instance.h"

using namespace std;

void Instance::readInputFile(const string& problemDir){
    ifstream file(problemDir);
    string line;

    while(getline(file,line)){
        istringstream iss(line);
        vector<string> values;


        string token;
        while (getline(iss, token, ',')) {
            values.push_back(token);
        }

        if(values.size() >= 6){
            int cusID = stoi(values[0]);
            demand[cusID] = stod(values[1]);
            pickup[cusID] = stod(values[2]);
            startTime[cusID] = stod(values[3]);
            endTime[cusID] = stod(values[4]);
            serviceTime[cusID] = stod(values[5]);
        }else{
            int c1 = stoi(values[0]);
            int c2 = stoi(values[1]);
            distMatrix[c1][c2] = stod(values[2]);
            timeMatrix[c1][c2] = stod(values[3]);
        }
    }
    file.close();
}

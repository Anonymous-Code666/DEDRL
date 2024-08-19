#ifndef INSTANCE_H
#define INSTANCE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// using namespace std;

class Instance {
public:
    int customerSize;
    std::vector<double> demand;
    std::vector<double> pickup;
    std::vector<double> startTime;
    std::vector<double> endTime;
    std::vector<double> serviceTime;
    std::vector<std::vector<double> > distMatrix;
    std::vector<std::vector<double> > timeMatrix;

    Instance(int size): customerSize(size),demand(size+1),pickup(size+1),
    startTime(size+1),endTime(size+1),serviceTime(size+1),
    distMatrix(size+1,std::vector<double>(size+1)),
    timeMatrix(size+1,std::vector<double>(size+1)){}

    void readInputFile(const std::string& problemDir);
};

#endif //INSTANCE_H

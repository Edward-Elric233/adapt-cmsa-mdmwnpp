//
// Created by edward on 9/25/24.
//

#include "SolutionSet.h"

using namespace std;

SolutionSet::SolutionSet(const std::string & results_file_path) {
    ifstream ifs(results_file_path);
    if (!ifs) {
        throw runtime_error("Can't read file: " + results_file_path);
    }
    string line, field;
    getline(ifs, line); //read table header
    while (getline(ifs, line)) {
        Solution solution(line);
        auto param = solution.getParam();
        param2solution_.insert({param, std::move(solution)});
    }
}

int SolutionSet::merge(const SolutionSet& other) {
    int cnt = 0;
    for (auto &&[param, otherSolution] : other.param2solution_) {
        if (auto iter = param2solution_.find(param); iter != param2solution_.end()) {
            auto &&solution = iter->second;
            if (otherSolution < solution) {
                ++cnt;
                solution = otherSolution;
            }
        } else {
            ++cnt;
            param2solution_.insert({param, otherSolution});
        }
    }
    return cnt;
}

std::ostream& operator<< (std::ostream &os, const SolutionSet& solutionSet) {
    for (auto &&[param, solution] : solutionSet.param2solution_) {
        os << solution << "\n";
    }
    return os;
}

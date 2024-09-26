//
// Created by edward on 9/25/24.
//

#include "Results.h"

#include <filesystem>
#include <fstream>

using namespace std;

Results::Results(const std::string &results_dir) {
    for (auto&& file_entry : std::filesystem::directory_iterator(results_dir)) {
        auto&& file_path = file_entry.path();
        if (filesystem::is_regular_file(file_entry.status()) && file_path.extension() == ".csv") {
            //结果文件
            auto&& data_set_name = file_path.stem().string();
            data2st_.insert({data_set_name, SolutionSet(file_path.string())});
        }
    }
}


int Results::merge(const Results& other) {
    int cnt = 0;
    for (auto &&[data_set_name, otherSolutionSet] : other.data2st_) {
        if (auto iter = data2st_.find(data_set_name); iter != data2st_.end()) {
            auto&& solutionSet = iter->second;
            cnt += solutionSet.merge(otherSolutionSet);
        } else {
            cnt += otherSolutionSet.param2solution_.size();
            data2st_.insert({data_set_name, otherSolutionSet});
        }
    }
    return cnt;
}

void Results::store(const std::string& store_results_dir) const {
    for (auto &&[data_set_name, solutionSet] : data2st_) {
        auto file_name = store_results_dir + data_set_name + ".csv";
        ofstream ofs(file_name);
        ofs << std::setprecision(4) << std::fixed;
        ofs << "n, m, k, value, time, valid, solution\n";
        ofs << solutionSet;
    }
}

std::ostream& operator<< (std::ostream &os, const Results& Results) {
    for (auto &&[data_set_name, solutionSet] : Results.data2st_) {
        os << "data set [" << data_set_name << "]:\n";
        os << solutionSet << "\n";
    }
    return os;
}

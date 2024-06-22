//
// Created by edward on 6/15/24.
//

#include <cstdio>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <unordered_map>
#include <sstream>
#include <vector>
#include <filesystem>
#include <sstream>
#include <numeric>
#include <algorithm>

using namespace std;

string run_command(string cmd) {
    constexpr int kBufSize = 1024;
    array<char, kBufSize> buffer;
    string result;
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw runtime_error("popen() for " + cmd + " failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

struct Param {
    int tILP;
    double aLB, aUB, aRed, tProp;
    Param() = default;
    explicit Param(int tILP_, double aLB_, double aUB_, double aRed_, double tProp_)
    : tILP(tILP_), aLB(aLB_), aUB(aUB_), aRed(aRed_), tProp(tProp_) {}
};

const unordered_map<int, Param> params = {
        {50, Param(2, 0.518, 0.898, 0.093, 0.557)},
        {100, Param(2, 0.302, 0.961, 0.049, 0.264)},
        {500, Param(6, 0.302, 0.948, 0.239, 0.069)},
};

const string work_dir = "/home/edward/code/cpp/adapt-cmsa-mdmwnpp/";
const string bin_path = work_dir + "source_codes/ADAPT_CMSA/mdmwnpp";
const string data_dir = work_dir + "instances/";
const unordered_map<string, string> data_file_names = {
        {"a", "mdtwnpp_500_20a.txt"},
        {"b", "mdtwnpp_500_20b.txt"},
        {"c", "mdtwnpp_500_20c.txt"},
        {"d", "mdtwnpp_500_20d.txt"},
        {"e", "mdtwnpp_500_20e.txt"},
};
const string results_dir = work_dir + "evaluate/results/";

struct Instance {
    vector<int> nSet, mSet, kSet;
    int t;
    Instance() = default;
    Instance(const vector<int> &nSet_, const vector<int> &mSet_, const vector<int> &kSet_, int t_)
    : nSet(nSet_), mSet(mSet_), kSet(kSet_), t(t_) {}
};

const vector<Instance> instances = {
        Instance({50, 100, 500}, {2, 5, 10, 20}, {2}, 600),
        Instance({50, 100}, {2, 3, 4, 5, 10, 15, 20}, {3, 4}, 1200),
        Instance({50, 100, 500}, {2, 5, 10, 20}, {5, 10, 20}, 1800),
};

template<typename... Args>
string join_fields(Args&&... args) {
    ostringstream oss;
    ((oss << args << " "), ...);
    return oss.str();
}

void run_single_instance() {
    string file_path = data_dir + data_file_names.at("a");
    int n = 50;
    int m = 4;
    int k = 3;
    int t = 120;
    auto&& param = params.at(n);
    auto cmd = join_fields(bin_path, "-f", file_path, "-n", n, "-m", m, "-k", k, "-alg", 5,
                        "-cmsa_cplex_time", 3, "-cmsa_greedy", 2, "-cmsa_milp", 0, "-n_a", 1, "-alphaLB", param.aLB, "-alphaUB", param.aUB,
                        "-alpha_red", param.aRed, "-t_prop", param.tProp, "-t", t);
    cout << cmd << endl;

    auto &&output = run_command(cmd);
    cout << output << endl;
}

void run_instances() {
    for (auto &&[set_name, file_name] : data_file_names) {
        string file_path = data_dir + file_name;
        const string result_file = results_dir + set_name + ".csv";
        {
            ofstream ofs(result_file, ios::app);
            if (!ofs) {
                throw runtime_error("Can't open file: " + result_file);
            }
            ofs << "n, m, k, value, time, valid, solution" << endl;
            ofs.close();
        }
        for (auto &&instance : instances) {
            for (auto &&n : instance.nSet) {
                for (auto &&m : instance.mSet) {
                    for (auto &&k : instance.kSet) {
                        auto&& param = params.at(n);
                        auto cmd = join_fields(bin_path, "-f", file_path, "-n", n, "-m", m, "-k", k, "-alg", 5,
                                            "-cmsa_cplex_time", 3, "-cmsa_greedy", 2, "-cmsa_milp", 0, "-n_a", 1, "-alphaLB", param.aLB, "-alphaUB", param.aUB,
                                            "-alpha_red", param.aRed, "-t_prop", param.tProp, "-t", instance.t);
                        cout << cmd << endl;
                        auto &&output = run_command(cmd);
                        ofstream ofs(result_file, ios::app);
                        if (!ofs) {
                            throw runtime_error("Can't write file: " + result_file);
                        }
                        ofs << output << flush;
                        ofs.close();
                    }
                }
            }
        }
    }
}

auto read_data(const string &data_file_path) {
    ifstream ifs(data_file_path);
    if (!ifs) {
        throw runtime_error("Can't read file: " + data_file_path);
    }
    int nMax, mMax;
    ifs >> nMax >> mMax;
    vector<vector<double>> vectors(nMax, vector<double>(mMax));
    for (int i = 0; i < nMax; ++i) {
        auto&& v = vectors[i];
        for (int j = 0; j < mMax; ++j) {
            ifs >> v[j];
        }
    }
    return vectors;
}

void check4solution(const vector<vector<double>> &vectors, int n, int m, int k, double value,
                    const vector<int> &solution) {
    constexpr double kEPS = 1e-3;
    if (solution.size() != n) {
        throw runtime_error("Solution size != n");
    }
    vector<vector<int>> partition(k);
    for (int i = 0; i < n; ++i) {
        partition[solution[i]].push_back(i);
    }
    for (int j = 0; j < m; ++j) {
        vector<double> sums(k, 0);
        for (int s = 0; s < k; ++s) {
            for (auto i : partition[s]) {
                sums[s] += vectors[i][j];
            }
        }
        sort(sums.begin(), sums.end());
        double diff = sums.back() - sums[0];
        if (diff > value + kEPS) {
            throw runtime_error(join_fields("Max diff on col", j , "is", diff, ", larger than", value));
        }
    }
    cout << "success" << endl;
}

void check4file(const string& result_file_path, const string& data_file_path) {
    auto&& vectors = read_data(data_file_path);
    ifstream ifs(result_file_path);
    if (!ifs) {
        throw runtime_error("Can't read file: " + result_file_path);
    }
    string line, field;
    getline(ifs, line); //read table header
    while (getline(ifs, line)) {
        istringstream line_stream(line);
        //n, m, k, value, time, valid, solution
        int n, m, k, x;
        double value;
        vector<int> solution;
        getline(line_stream, field, ',');   n = stoi(field);
        getline(line_stream, field, ',');   m = stoi(field);
        getline(line_stream, field, ',');   k = stoi(field);
        getline(line_stream, field, ',');   value = stod(field);
        getline(line_stream, field, ',');   //time
        getline(line_stream, field, ',');   //valid
        solution.reserve(n);
        while (line_stream >> x) {
            solution.push_back(x);
        }
        try {
            check4solution(vectors, n, m, k, value, solution);
        } catch (const exception &e) {
            std::cerr << "Error: " << join_fields(result_file_path, n, m, k, ":", e.what()) << std::endl;
        }
    }
}

void check() {  //检查运行结果正确性
    for (auto&& file_entry : filesystem::directory_iterator(results_dir)) {
        auto&& file_path = file_entry.path();
        if (filesystem::is_regular_file(file_entry.status()) && file_path.extension() == ".csv") {
            //结果文件
            auto&& data_set_name = file_path.stem().string();
            check4file(file_path.string(), data_dir + data_file_names.at(data_set_name));
        }
    }
}

int main() {
    try {
//        run_single_instance();
//        run_instances();
        check();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
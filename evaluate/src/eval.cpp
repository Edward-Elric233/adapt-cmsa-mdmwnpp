//
// Created by edward on 6/15/24.
//

#include "utils.h"
#include "Results.h"

#include <boost/program_options.hpp>

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

namespace po = boost::program_options;
using namespace std;

//TODO: move to utils
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
const string bin_path = work_dir + "source_codes/ADAPT_CMSA/bazel-bin/mdmwnpp";
const string data_dir = work_dir + "instances/";
const unordered_map<string, string> data_file_names = {
        {"a", "mdtwnpp_500_20a.txt"},
//        {"b", "mdtwnpp_500_20b.txt"},
//        {"c", "mdtwnpp_500_20c.txt"},
//        {"d", "mdtwnpp_500_20d.txt"},
//        {"e", "mdtwnpp_500_20e.txt"},
};

const string results_dir = work_dir + "evaluate/results/";
const string cur_results_dir = results_dir + "cur/";

struct Instance {
    vector<int> nSet, mSet, kSet;
    int t;
    Instance() = default;
    Instance(const vector<int> &nSet_, const vector<int> &mSet_, const vector<int> &kSet_, int t_)
    : nSet(nSet_), mSet(mSet_), kSet(kSet_), t(t_) {}
};

//TODO: move to config
const vector<Instance> instances = {
//        Instance({50}, {2}, {2}, 600),


        Instance({50, 100, 500}, {2, 5, 10, 20}, {2}, 600),   //2h


//        Instance({50, 100}, {2, 3, 4, 5, 10, 15, 20}, {3, 4}, 1200),
//        Instance({50, 100}, {10, 15, 20}, {3, 4}, 1200),
//        Instance({50, 100}, {2, 3, 4, 5}, {3, 4}, 1200),      //5.3h


//        Instance({50, 100, 500}, {2, 5, 10, 20}, {5, 10, 20}, 1800),
        Instance({50}, {2, 5, 10, 20}, {5, 10, 20}, 1800),      //6h
//        Instance({100}, {2, 5, 10, 20}, {5, 10, 20}, 1800),
//        Instance({500}, {2, 5, 10, 20}, {5, 10, 20}, 1800),
};


void run_single_instance() {
    string file_path = data_dir + data_file_names.at("a");
    int n = 50;
    int m = 4;
    int k = 3;
    int t = 120;
    auto&& param = params.at(n);
    auto cmd = edward::join_fields(bin_path, "-f", file_path, "-n", n, "-m", m, "-k", k, "-alg", 5,
                        "-cmsa_cplex_time", 3, "-cmsa_greedy", 2, "-cmsa_milp", 0, "-n_a", 1, "-alphaLB", param.aLB, "-alphaUB", param.aUB,
                        "-alpha_red", param.aRed, "-t_prop", param.tProp, "-t", t);
    cout << cmd << endl;

    auto &&output = run_command(cmd);
    cout << output << endl;
}

void run_instances() {
    for (auto &&[set_name, file_name] : data_file_names) {
        string file_path = data_dir + file_name;
        const string result_file = cur_results_dir + set_name + ".csv";
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
                        auto cmd = edward::join_fields(bin_path, "-f", file_path, "-n", n, "-m", m, "-k", k, "-alg", 5,
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
    if (static_cast<int>(solution.size()) != n) {
        throw runtime_error("Solution size != n");
    }
    vector<vector<int>> partition(k);
    for (int i = 0; i < n; ++i) {
        partition[solution[i]].push_back(i);
    }
    double objective = numeric_limits<double>::min();
    for (int j = 0; j < m; ++j) {
        vector<double> sums(k, 0);
        for (int s = 0; s < k; ++s) {
            for (auto i : partition[s]) {
                sums[s] += vectors[i][j];
            }
        }
        sort(sums.begin(), sums.end());
        double diff = sums.back() - sums[0];
        objective = std::max(diff, objective);
    }
    if (objective > value + kEPS) {
        throw runtime_error(edward::join_fields("Max diff", "is", objective, ", larger than", value));
    } else if (objective < value - kEPS) {
        throw runtime_error(edward::join_fields("Max diff", "is", objective, ", smaller than", value));
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
            std::cerr << "Error: " << edward::join_fields(result_file_path, n, m, k, ":", e.what()) << std::endl;
        }
    }
}

void check() {  //检查运行结果正确性
    for (auto&& file_entry : filesystem::directory_iterator(cur_results_dir)) {
        auto&& file_path = file_entry.path();
        if (filesystem::is_regular_file(file_entry.status()) && file_path.extension() == ".csv") {
            //结果文件
            auto&& data_set_name = file_path.stem().string();
            check4file(file_path.string(), data_dir + data_file_names.at(data_set_name));
        }
    }
}

void store(const string &src_results_dir, const string &dst_results_dir, const string &store_results_dir) {
    Results srcResults(src_results_dir);
//    cout << "src results\n" << srcResults << endl;
    Results dstResults(dst_results_dir);
//    cout << "dst results\n" << dstResults << endl;
    int cnt = dstResults.merge(srcResults);
    cout << "src has " << cnt << " solutions better than dst" << endl;
    if (store_results_dir.size() > 0) {
        try {
            //remove all csv file
            for (auto&& file_entry : filesystem::directory_iterator(store_results_dir)) {
                if (file_entry.is_regular_file() && file_entry.path().extension() == ".csv") {
                    filesystem::remove(file_entry.path());
                }
            }

            dstResults.store(store_results_dir);
        } catch (const filesystem::filesystem_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("run,r", "run all instances declared")
            ("check,c", "check results availability")
            ("store,s", po::value<vector<string>>()->multitoken()->zero_tokens(), "store best results to results-best")
            ("all,a", "-r -c -s");
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        if (vm.count("run") || vm.count("all")) {
            //TODO: check cur dir
            run_instances();
        }
        if (vm.count("check") || vm.count("all")) {
            check();
        }
        if (vm.count("store") || vm.count("all")) {
            vector<string> store_options;
            if (vm.count("store")) {
                store_options = vm["store"].as<vector<string>>();
            }
            if (store_options.size() > 3) {
                std::cerr << "Option 'store' accepts at most 3 arguments.\n";
                return 0;
            }
            string src_results = "cur";
            string dst_results = "best";
            string dst_dir; //default is empty
            if (store_options.size() > 0) {
                src_results = store_options[0];
            }
            if (store_options.size() > 1) {
                dst_results = store_options[1];
            }
            if (store_options.size() > 2) {
                dst_dir = results_dir + store_options[2] + "/";
            }
            store(results_dir + src_results, results_dir + dst_results, dst_dir);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
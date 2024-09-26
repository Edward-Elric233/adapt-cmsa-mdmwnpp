//
// Created by edward on 9/23/24.
//

#include "utils.h"
#include "Solution.h"

using namespace std;

hash<int> hash<SolutionParam>::helper;

bool SolutionParam::operator== (const SolutionParam& other) const {
    return n_ == other.n_ && m_ == other.m_ && k_ == other.k_;
}
bool SolutionParam::operator< (const SolutionParam& other) const {
    return n_ < other.n_ || (n_ == other.n_ && m_ < other.m_) || (n_ == other.n_ && m_ == other.m_ && k_ < other.k_);
}


Solution::Solution(const std::string &line) {
    istringstream line_stream(line);
    string field;
    getline(line_stream, field, ',');   param_.n_ = stoi(field);
    getline(line_stream, field, ',');   param_.m_ = stoi(field);
    getline(line_stream, field, ',');   param_.k_ = stoi(field);
    getline(line_stream, field, ',');   value_ = stod(field);
    getline(line_stream, field, ',');   time_ = stod(field);
    getline(line_stream, field, ',');   valid_ = static_cast<bool>(stoi(field));
    solution_.resize(param_.n_);
    for (auto &&x : solution_) {
        line_stream >> x;
    }
}

const SolutionParam& Solution::getParam() const {
    return param_;
}


bool Solution::operator< (const Solution& other) const {
    return value_ < other.value_;
}

std::ostream& operator<< (std::ostream &os, const Solution& solution) {
    os << solution.param_.n_ << ", "
        << solution.param_.m_ << ", "
        << solution.param_.k_ << ", "
        << solution.value_ << ", "
        << solution.time_ << ", "
        << static_cast<int>(solution.valid_) << ", ";
    for (auto x : solution.solution_) {
        os << x << " ";
    }
    return os;
}
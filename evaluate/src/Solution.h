//
// Created by edward on 9/23/24.
//

#ifndef ADAPT_CMSA_MDMWNPP_SOLUTION_H
#define ADAPT_CMSA_MDMWNPP_SOLUTION_H

#include <vector>
#include <string>
#include <sstream>

struct SolutionParam {
    int n_, m_, k_;
    bool operator== (const SolutionParam& other) const;
    bool operator< (const SolutionParam& other) const;
};

namespace std {
    template<>
    struct hash<SolutionParam> {
        static hash<int> helper;
        using result_type = size_t;
        using argument_type = SolutionParam;
        result_type operator() (const argument_type& x) const {
            auto h1 = helper(x.n_);
            auto h2 = helper(x.m_);
            auto h3 = helper(x.k_);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

class Solution {
    SolutionParam param_;
    double value_, time_;
    bool valid_;
    std::vector<int> solution_;
public:
    explicit Solution(const std::string& line);
    const SolutionParam& getParam() const;
    bool operator< (const Solution& other) const;

    friend std::ostream& operator<< (std::ostream &os, const Solution& solution);
};


#endif //ADAPT_CMSA_MDMWNPP_SOLUTION_H

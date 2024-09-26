//
// Created by edward on 9/25/24.
//

#ifndef ADAPT_CMSA_MDMWNPP_RESULTS_H
#define ADAPT_CMSA_MDMWNPP_RESULTS_H

#include "SolutionSet.h"

#include <string>
#include <unordered_map>

class Results {
    std::unordered_map<std::string, SolutionSet> data2st_;
public:
    explicit Results(const std::string &results_dir);
    int merge(const Results& other);
    void store(const std::string& store_results_dir) const;

    friend std::ostream& operator<< (std::ostream &os, const Results& Results);
};


#endif //ADAPT_CMSA_MDMWNPP_RESULTS_H

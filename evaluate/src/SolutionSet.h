//
// Created by edward on 9/25/24.
//

#ifndef ADAPT_CMSA_MDMWNPP_SOLUTIONSET_H
#define ADAPT_CMSA_MDMWNPP_SOLUTIONSET_H

#include "Solution.h"

#include <string>
#include <unordered_map>
#include <fstream>
#include <map>

class SolutionSet {
    std::map<SolutionParam, Solution> param2solution_;
public:
    explicit SolutionSet(const std::string& results_file_path);
    int merge(const SolutionSet& other);

    friend class Results;
    friend std::ostream& operator<< (std::ostream &os, const SolutionSet& solutionSet);
};


#endif //ADAPT_CMSA_MDMWNPP_SOLUTIONSET_H

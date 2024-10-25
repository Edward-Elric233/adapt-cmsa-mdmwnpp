// Copyright(C), Edward-Elric233
// Author: Edward-Elric233
// Version: 1.0
// Date: 2022/6/27
// Description:
#include "utils.h"

namespace edward {

     Random::random_engine_type Random::pseudoRandNumGen(std::chrono::system_clock::now().time_since_epoch().count()); //默认使用当下时间戳初始化随机数种子，精确到纳秒

}

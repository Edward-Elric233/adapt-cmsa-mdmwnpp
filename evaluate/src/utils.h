// Copyright(C), Edward-Elric233
// Author: Edward-Elric233
// Version: 1.0
// Date: 2022/6/27
// Description:
#ifndef UTILS_H
#define UTILS_H

#include "json.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>

namespace edward {

    constexpr int INF = 0x3f3f3f3f;

    template<typename... Args>
    std::string join_fields(Args&&... args) {
        std::ostringstream oss;
        ((oss << args << " "), ...);
        return oss.str();
    }

    template<typename... Args>
    void print(Args&&... args) {
        ((std::cout << args << " "), ...);
        std::cout << std::endl;
    }

    template<typename Iter>
    void printArr(Iter begin, Iter end) {
        while (begin != end) {
            std::cout << *begin++ << " ";
        }
        std::cout << "\n";
    }

    template<typename T1, typename T2>
    std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& pr) {
        os << pr.first << " " << pr.second;
        return os;
    }

    template<typename Container>
    void printArr(const Container& container) {
        for (auto x : container) {
            std::cout << x << " ";
        }
        std::cout << "\n";
    }

    class Random {
    public:
        // random number generator.
        static std::mt19937 pseudoRandNumGen;
        static void initRand(int seed) { pseudoRandNumGen = std::mt19937(seed); }   //设置随机数种子
        static int fastRand(int lb, int ub) { return (pseudoRandNumGen() % (ub - lb)) + lb; }
        static int fastRand(int ub) { return pseudoRandNumGen() % ub; }
        static int rand(int lb, int ub) { return std::uniform_int_distribution<int>(lb, ub - 1)(pseudoRandNumGen); }
        static int rand(int ub) { return std::uniform_int_distribution<int>(0, ub - 1)(pseudoRandNumGen); }
    };

    class Timer {
        std::chrono::time_point<std::chrono::system_clock> timePoint_;
    public:
        Timer(): timePoint_(std::chrono::system_clock::now()) {}
        Timer(const Timer&) = delete;
        ~Timer() = default;
        void operator() (const std::string& msg) {
            auto duration = std::chrono::system_clock::now() - timePoint_;
            print(msg, static_cast<double>(duration.count()) / decltype(duration)::period::den, "s");
        }
        void reset() {
            timePoint_ = std::chrono::system_clock::now();
        }
        void operator() (nlohmann::json& arr) {
            auto duration = std::chrono::system_clock::now() - timePoint_;
            arr.push_back(duration.count());
        }

    };



    using Slot = std::shared_ptr<void>;

//前置声明
    template<typename Signature>
    class Signal;
    template<typename Ret, typename... Args>
    class Signal<Ret(Args...)>;

    namespace detail {
        //前置声明
        template<typename Callback> class SlotImpl;

        template<typename Callback>
        class SignalImpl {
        public:
            using SlotList = std::unordered_map<SlotImpl<Callback> *, std::weak_ptr<SlotImpl<Callback>>>;

            SignalImpl()
                    : slots_(new SlotList)
                    , mutex_() {

            }
            ~SignalImpl() {

            }

            //只能在加锁后使用
            void cowWithLock() {
                if (!slots_.unique()) {
                    slots_.reset(new SlotList(*slots_));
                }
            }

            //添加槽函数
            void add(const std::shared_ptr<SlotImpl<Callback>> &slot) {
                std::lock_guard<std::mutex> lockGuard(mutex_);
                cowWithLock();
                slots_->insert({slot.get(), slot});
            }
            //供SlotImpl调用，删除槽函数
            void remove(SlotImpl<Callback> *slot) {
                std::lock_guard<std::mutex> lockGuard(mutex_);
                cowWithLock();
                slots_->erase(slot);
            }

            std::shared_ptr<SlotList> getSlotList() {
                std::lock_guard<std::mutex> lockGuard(mutex_);
                return slots_;
            }
        private:
            std::mutex mutex_;
            //保存SlotImpl的weak_ptr
            //之所以不保存SlotList而是保存其shared_ptr是为了实现COW
            std::shared_ptr<SlotList> slots_;
        };

        template<typename Callback>
        class SlotImpl {
        public:
            SlotImpl(Callback&& cb, const std::shared_ptr<SignalImpl<Callback>> &signal)
                    : cb_(cb)
                    , signal_(signal) {

            }
            ~SlotImpl() {
                auto signal = signal_.lock();
                if (signal) {
                    signal->remove(this);
                }
            }

            Callback cb_;
        private:
            //保存SignalImpl的weak_ptr
            std::weak_ptr<SignalImpl<Callback>> signal_;
        };
    }

    template<typename Ret, typename... Args>
    class Signal<Ret(Args...)> {
    public:
        using Callback = std::function<Ret(Args...)>;
        using SignalImpl = detail::SignalImpl<Callback>;
        using SlotImpl = detail::SlotImpl<Callback>;

        Signal()
                : impl_(new SignalImpl) {

        }

        ~Signal() {

        }

        /*!
         * 添加槽函数
         * @param cb 槽函数
         * @return 需要保存这个智能指针，否则会自动从槽函数列表中删除
         */
        template<typename Func>
        Slot connect(Func&& cb) {
            std::shared_ptr<SlotImpl> slot(new SlotImpl(std::forward<Func>(cb), impl_));
            impl_->add(slot);
            return slot;
        }

        template<typename ...ARGS>
        void operator() (ARGS&&... args) {
            auto slots = impl_->getSlotList();
            //使用引用避免智能指针的解引用
            auto &s = *slots;
            for (auto &&[pSlotImpl, pWkSlotImpl] : s) {
                auto pShrSlotImpl = pWkSlotImpl.lock();
                if (pShrSlotImpl) {
                    pShrSlotImpl->cb_(std::forward<ARGS>(args)...);
                }
            }
        }

    private:
        //保存shared_ptr的原因是需要传递给SlotImpl，在SlotImpl析构的时候会清除自己
        const std::shared_ptr<SignalImpl> impl_;
    };



}

#endif //UTILS_H


//
// Created by edward on 9/13/24.
//

#ifndef ADAPT_CMSA_MDMWNPP_MATRIX_H
#define ADAPT_CMSA_MDMWNPP_MATRIX_H

#include <memory>
#include <stdexcept>
#include <cstring>
#include <cassert>

/*!
 * 按照行优先连续存储的二维数组
 */
template <typename T>
class Matrix {
    int n_, m_;
    T* p_ = nullptr;
public:
    Matrix(int n, int m)
    : n_(n), m_(m) {
        if (n <= 0 || m <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        p_ = new T[n_ * m_];
    }
    ~Matrix() {
        delete[] p_;
    }
    Matrix(const Matrix& other) = delete;
    void operator= (const Matrix& other) = delete;
    Matrix(Matrix&& other) noexcept
    : n_(other.n_), m_(other.m_), p_(other.p_) {
        other.p_ = nullptr;
        other.n_ = 0;
        other.m_= 0 ;
    }
    Matrix& operator= (Matrix&& other) noexcept {
        if (this != &other) {
            delete[] p_;
            n_ = other.n_;
            m_ = other.m_;
            p_ = other.p_;
            other.p_ = nullptr;
            other.n_ = 0;
            other.m_= 0 ;
        }
        return *this;
    }
    T& operator() (int x, int y) {
        assert(x >= 0 && x < n_ && y >= 0 && y < m_);
        return p_[x * m_ + y];
    }
    const T& operator() (int x, int y) const {
        assert(x >= 0 && x < n_ && y >= 0 && y < m_);
        return p_[x * m_ + y];
    }
    int getRows() const {
        return n_;
    }
    int getCols() const {
        return m_;
    }
    T* operator[] (int x) {
        assert(x >= 0 && x < n_);
        return p_ + x * m_;
    }
    void reset() {
        assert(p_ != nullptr);
        if constexpr (std::is_trivially_default_constructible<T>::value) {
            // 对于平凡类型，使用 memset
            std::memset(p_, 0, n_ * m_ * sizeof(T));
        } else {
            // 对于非平凡类型，使用默认构造函数
            for (int i = 0; i < n_ * m_; ++i) {
                p_[i] = T();
            }
        }
    }
};

#endif //ADAPT_CMSA_MDMWNPP_MATRIX_H

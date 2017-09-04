#ifndef TENSORFLOW_FOLD_LOOM_HASHES_H_
#define TENSORFLOW_FOLD_LOOM_HASHES_H_
// This file contains specializations of std::hash.

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace {

template<class T, class Hash = std::hash<T>>
void hash_combine(std::size_t* seed, const T& other) {
  *seed ^= Hash{}(other) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct TupleHash {
  size_t operator()(const Tuple& tuple) const {
    size_t seed = TupleHash<Tuple, Index - 1>{}(tuple);
    hash_combine(&seed, std::get<Index>(tuple));
    return seed;
  }
};

template <class Tuple>
struct TupleHash<Tuple, 0> {
  size_t operator()(const Tuple& tuple) const {
    return std::hash<typename std::tuple_element<0, Tuple>::type>{}(
        std::get<0>(tuple));
  }
};

}

namespace std {

template <class T1, class T2>
struct hash<pair<T1, T2>> {
  size_t operator()(const pair<T1, T2> &pair) const {
    size_t seed = hash<T1>{}(pair.first);
    hash_combine(&seed, pair.second);
    return seed;
  }
};

template <class... TT>
struct hash<tuple<TT...>> {
  size_t operator()(const tuple<TT...>& tuple) const {
    return TupleHash<std::tuple<TT...> >()(tuple);
  }
};

template <class T>
struct hash<vector<T>> {
  size_t operator()(const vector<T> &v) const {
    size_t seed = 0;
    for (const auto& it : v) {
      hash_combine(&seed, it);
    }
    return seed;
  }
};

template<>
struct hash<tensorflow::TensorShape> {
  size_t operator()(const tensorflow::TensorShape& ts) const {
    size_t seed = std::hash<int>{}(ts.dims());
    for (int d = 0; d < ts.dims(); ++d) {
      hash_combine(&seed, ts.dim_size(d));
    }
    return seed;
  }
};

template <>
struct hash<tensorflow::Tensor> {
  size_t operator()(const tensorflow::Tensor& t) const {
    size_t seed = std::hash<tensorflow::TensorShape>{}(t.shape());
    hash_combine<tensorflow::StringPiece, tensorflow::StringPiece::Hasher>(
        &seed, t.tensor_data());
    return seed;
  }
};

}  // namespace std

#endif

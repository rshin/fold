/* Copyright 2017 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_fold/loom/weaver.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>

#include "tensorflow_fold/loom/loom.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"

#define LOOM_DEDUP_DEBUG 0

#ifdef LOOM_DEDUP_DEBUG
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}
}
#endif

namespace tensorflow {
namespace fold {

using tensorflow::strings::StrCat;

namespace {

// `FastBoundsCheck` is copied from tensorflow/core/kernels/bounds_check.h due
// to visibility constraints.
//
// Check that 0 <= index < limit using a single comparison, assuming
// that 0 <= limit if Index is signed.  Intended for use in performance
// critical contexts where 0 <= index < limit is almost always true.
template <typename Ta, typename Tb>
bool FastBoundsCheck(const Ta index, const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                         static_cast<UIndex>(limit));
}

Tensor StackTensors(const std::vector<Tensor> &tensors,
                    DataType dtype,
                    const std::vector<int64> &shape) {
  if (tensors.empty()) {
    TensorShape empty_shape(shape);
    empty_shape.InsertDim(0, 0);
    return Tensor(dtype, empty_shape);
  }

  std::vector<Tensor> expanded_tensors;
  TensorShape expanded_shape(shape);
  expanded_shape.InsertDim(0, 1);
  for (const auto & t : tensors) {
    expanded_tensors.emplace_back(dtype, expanded_shape);
    CHECK(expanded_tensors.back().CopyFrom(t, expanded_shape))
        << "Failed to reshape tensor during StackTensors.";
  }
  tensorflow::Tensor concated;
  CHECK(tensorflow::tensor::Concat(expanded_tensors, &concated).ok());
  return concated;
}

std::vector<Tensor> UnstackTensors(const Tensor &stacked) {
  TensorShape shape = stacked.shape();
  shape.RemoveDim(0);
  std::vector<int64> sizes(stacked.shape().dim_size(0), 1);
  std::vector<Tensor> split;
  CHECK(tensorflow::tensor::Split(stacked, sizes, &split).ok());
  std::vector<Tensor> result;
  for (const Tensor &t : split) {
    result.emplace_back(t.dtype(), shape);
    CHECK(result.back().CopyFrom(t, shape))
        << "Failed to reshape tensor during UnstackTensors.";
  }
  return result;
}

LoomResult LoomResultFromMessage(
    const WeaverMessage &message, int i) {
  LoomResult r;
  r.depth = message.depth(i);
  r.ts_idx = message.ts_idx(i);
  r.op_idx = message.op_idx(i);
  r.op_output_idx = message.op_output_idx(i);
  r.pos_idx = message.pos_idx(i);
  r.cached_passthrough = message.cached_passthrough(i);
  return r;
}

template <typename T>
void enlarge(std::vector<T>* vec, tensor_idx_t idx) {
  if (vec->size() <= idx) {
    vec->resize(idx + 1);
  }
}

}  // namespace

bool VerifyLoomMetadata(const LoomMetadata &metadata, string *error_string) {
  if (metadata.max_depth() < -1) {
    *error_string = StrCat(
        "metadata.max_depth must be -1 or greater.  Got: ",
        metadata.max_depth());
    return false;
  }

  tensor_idx_t num_type_shapes = metadata.type_shape_metadata_size();
  tensor_idx_t num_ops = metadata.op_metadata_size();

  for (tensor_idx_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    const auto& op = metadata.op_metadata(op_idx);
    if (op.input_ts_idx_size() <= 0) {
      *error_string = StrCat(
          "metadata.op_metadata[", op_idx,
          "] should have at least one input.");
      return false;
    }
    for (tensor_idx_t ts_idx : op.input_ts_idx()) {
      if (!FastBoundsCheck(ts_idx, num_type_shapes)) {
        *error_string = StrCat(
            "metadata.op_metadata[", op_idx,
            "] has an invalid input typeshape: ", ts_idx);
        return false;
      }
    }

    if (op.output_ts_idx_size() <= 0) {
      *error_string = StrCat(
          "metadata.op_metadata[", op_idx,
          "] should have at least one output.");
      return false;
    }
    for (tensor_idx_t ts_idx : op.output_ts_idx()) {
      if (!FastBoundsCheck(ts_idx, num_type_shapes)) {
        *error_string = StrCat(
            "metadata.op_metadata[", op_idx,
            "] has an invalid output typeshape: ", ts_idx);
        return false;
      }
    }
  }
  if (metadata.use_tensor_array()) {
    // No need to look for passthrough ops.
    return true;
  }

  if (num_type_shapes > num_ops) {
    *error_string = StrCat(
        "metadata specifies num_type_shapes (", num_type_shapes,
        ") which ought to be at most num_ops (", num_ops, ")");
    return false;
  }

  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes; ++ts_idx) {
    const auto& op = metadata.op_metadata(ts_idx);
    if (op.input_ts_idx_size() != 1) {
      *error_string = StrCat(
        "PassThrough Op ", ts_idx, " ought to have a single input.");
      return false;
    }
    if (op.output_ts_idx_size() != 1) {
      *error_string = StrCat(
        "PassThrough Op ", ts_idx, " ought to have a single output.");
      return false;
    }
    if (op.input_ts_idx(0) != ts_idx) {
      *error_string = StrCat(
        "PassThrough Op ", ts_idx, " input has the wrong TypeShape:",
        op.input_ts_idx(0));
      return false;
    }
    if (op.output_ts_idx(0) != ts_idx) {
      *error_string = StrCat(
        "PassThrough Op ", ts_idx, " output has the wrong TypeShape:",
        op.output_ts_idx(0));
      return false;
    }
  }
  return true;
}

Weaver::Weaver(const string &serialized_loom_metadata) {
  if (!metadata_.ParseFromString(serialized_loom_metadata)) {
    error_string_ = "Could not parse metadata";
    return;
  }

  if (!VerifyLoomMetadata(metadata_, &error_string_)) return;

  max_depth_ = metadata_.max_depth();
  use_tensor_array_ = metadata_.use_tensor_array();
  deduplicate_ = metadata_.deduplicate();

  num_type_shapes_ = metadata_.type_shape_metadata_size();
  type_shapes_.resize(num_type_shapes_);
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    const auto& ts_metadata = metadata_.type_shape_metadata(ts_idx);
    std::vector<int64> shape(ts_metadata.shape().begin(),
                             ts_metadata.shape().end());
    type_shapes_[ts_idx].dtype = ts_metadata.dtype();
    type_shapes_[ts_idx].shape = tensorflow::TensorShape(shape);
    type_shapes_[ts_idx].name = ts_metadata.name();

    num_named_tensors_by_ts_idx_.emplace_back(ts_metadata.tensor_names_size());
  }

  num_ops_ = metadata_.op_metadata_size();
  for (const auto &op : metadata_.op_metadata()) {
    op_names_.emplace_back(op.name());
    op_input_ts_idx_.emplace_back(
        op.input_ts_idx().begin(),
        op.input_ts_idx().end());
    op_output_ts_idx_.emplace_back(
        op.output_ts_idx().begin(),
        op.output_ts_idx().end());
  }

  deepest_ = 0;
  num_constants_by_type_shape_.resize(num_type_shapes_);
  constant_values_by_type_shape_.resize(num_type_shapes_);
  finalized_ = false;
}

void Weaver::Reset() {
  finalized_ = false;
  deepest_ = 0;
  fill(num_constants_by_type_shape_.begin(),
       num_constants_by_type_shape_.end(), 0);

  constant_values_by_type_shape_.clear();
  constant_values_by_type_shape_.resize(num_type_shapes_);

  loom_results_.clear();
  output_result_ids_.clear();
  wiring_results_.clear();
  op_output_wiring_results_.clear();
  final_wiring_.clear();
  num_results_per_type_shape_.clear();
  final_output_wiring_.clear();
}

Tensor Weaver::BatchConstantValues(tensor_idx_t ts_idx) const {
  std::vector<int64> shape(
      metadata_.type_shape_metadata(ts_idx).shape().begin(),
      metadata_.type_shape_metadata(ts_idx).shape().end());
  return StackTensors(
      constant_values_by_type_shape_[ts_idx],
      metadata_.type_shape_metadata(ts_idx).dtype(), shape);
}

string Weaver::Serialize() const {
  WeaverMessage message;

  for (const LoomResult &r : loom_results_) {
    message.add_depth(r.depth);
    message.add_ts_idx(r.ts_idx);
    message.add_op_idx(r.op_idx);
    message.add_op_output_idx(r.op_output_idx);
    message.add_pos_idx(r.pos_idx);
    message.add_cached_passthrough(r.cached_passthrough);
  }

  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_ ; ++ts_idx) {
    message.add_num_constants_by_type_shape(
        num_constants_by_type_shape_[ts_idx]);
    TensorProto *constants = message.add_constant_values_by_type_shape();
    Tensor stacked = BatchConstantValues(ts_idx);
    stacked.AsProtoTensorContent(constants);
  }

  for (const auto &pair : wiring_results_) {
    tensor_idx_t depth, op_idx, arg_idx;
    std::tie(depth, op_idx, arg_idx) = pair.first;
    auto *wiring = message.add_wiring();
    wiring->set_depth(depth);
    wiring->set_op_idx(op_idx);
    wiring->set_arg_idx(arg_idx);
    for (tensor_idx_t result_id : pair.second) {
      wiring->add_result_id(result_id);
    }
  }

  for (const auto &pair : op_output_wiring_results_) {
    tensor_idx_t depth, op_idx, arg_idx;
    std::tie(depth, op_idx, arg_idx) = pair.first;
    auto *wiring = message.add_op_output_wiring();
    wiring->set_depth(depth);
    wiring->set_op_idx(op_idx);
    wiring->set_arg_idx(arg_idx);
    for (tensor_idx_t result_id : pair.second) {
      wiring->add_result_id(result_id);
    }
  }

  for (tensor_idx_t result_id : output_result_ids_) {
    message.add_output_result_id(result_id);
  }

  if (deduplicate_) {
    for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
      message.add_constant_values_result_ids()
          ->mutable_result_id()
          ->Resize(num_constants_by_type_shape_[ts_idx], -1);
    }

    for (const auto& kv : cached_constant_values_) {
      const auto& ts_idx_and_tensor = kv.first;
      const CachedConstantValueEntry& entry = kv.second;

      message.mutable_constant_values_result_ids(ts_idx_and_tensor.first)
          ->set_result_id(entry.constant_value_idx, entry.result_id);
    }
  }

  string result;
  if (!message.SerializeToString(&result)) {
    error_string_ = "Serialization to WeaverMessage failed.  "
        "Did you run into the protocol buffer size limit?";
    return "";
  }
  return result;
}

bool Weaver::Deserialize(const string &weaver_message) {
  WeaverMessage message;
  if (!message.ParseFromString(weaver_message)) {
    error_string_ = "WeaverMessage couldn't be parsed.";
    return false;
  }

  if (num_constants_by_type_shape_.size() !=
      message.num_constants_by_type_shape_size()) {
    error_string_ =
        "WeaverMessage didn't have the expected number of type-shapes.";
    return false;
  }

  Reset();

  tensor_idx_t num_loom_results = message.depth_size();

  for (tensor_idx_t i = 0; i < num_loom_results; ++i) {
    loom_results_.emplace_back(LoomResultFromMessage(message, i));

    deepest_ = std::max(deepest_, loom_results_.back().depth);
  }

  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    num_constants_by_type_shape_[ts_idx] =
        message.num_constants_by_type_shape(ts_idx);
    Tensor constants(metadata_.type_shape_metadata(ts_idx).dtype());
    if (!constants.FromProto(message.constant_values_by_type_shape(ts_idx))) {
      error_string_ = StrCat(
          "Conversion from TensorProto to Tensor failed in deserialization.  ",
          "ts_idx=", ts_idx);
      return false;
    }
    constant_values_by_type_shape_[ts_idx] = UnstackTensors(constants);
  }

  for (const auto &w : message.wiring()) {
    std::vector<tensor_idx_t> result_ids(
        w.result_id().begin(), w.result_id().end());
    auto key = std::make_tuple(w.depth(), w.op_idx(), w.arg_idx());
    wiring_results_.emplace_hint(wiring_results_.end(), key, result_ids);
  }

  for (const auto &w : message.op_output_wiring()) {
    std::vector<tensor_idx_t> result_ids(
        w.result_id().begin(), w.result_id().end());
    auto key = std::make_tuple(w.depth(), w.op_idx(), w.arg_idx());
    op_output_wiring_results_.emplace_hint(op_output_wiring_results_.end(), key,
                                           result_ids);
  }

  output_result_ids_.insert(
      output_result_ids_.end(),
      message.output_result_id().begin(),
      message.output_result_id().end());

  if (deduplicate_) {
    // Fill cached_constant_values_.
    if (message.constant_values_result_ids_size() !=
        message.constant_values_by_type_shape_size()) {
      error_string_ = "WeaverMessage is missing constant_values_result_ids.";
      return false;
    }
    for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
      const auto &result_ids =
          message.constant_values_result_ids(ts_idx).result_id();
      for (tensor_idx_t i = 0; i < result_ids.size(); ++i) {
        CachedConstantValueEntry entry{result_ids[i], i, ts_idx};
        cached_constant_values_.emplace(
            std::make_pair(ts_idx, constant_values_by_type_shape_[ts_idx][i]),
            std::move(entry));
      }
    }

    // Fill cached_calls_.
    // We take advantage of how LoomResults representing the different outputs
    // of a single call are adjacent.
    tensor_idx_t depth = -1, op_idx = -1, pos_idx = -1;
    std::vector<tensor_idx_t> args, result_ids;
    for (tensor_idx_t i = 0; i < loom_results_.size(); ++i) {
      const LoomResult& r = loom_results_[i];
      if (r.depth < 1) {
        continue;
      }

#if LOOM_DEDUP_DEBUG
      std::cerr << "LR id=" << i << " depth=" << r.depth
                << " ts_idx=" << r.ts_idx << " op_idx=" << r.op_idx
                << " op_output_idx=" << r.op_output_idx
                << " pos_idx=" << r.pos_idx << std::endl;
#endif

      if (depth != r.depth || op_idx != r.op_idx || pos_idx != r.pos_idx) {
        if (!args.empty()) {
          CHECK_EQ(result_ids.size(), op_output_ts_idx_[op_idx].size());
#if LOOM_DEDUP_DEBUG
          std::cerr << "Adding op_idx=" << op_idx << " args=" << args
                    << " result_ids=" << result_ids << std::endl;
#endif
          cached_calls_.emplace(std::make_pair(op_idx, args), result_ids);
          result_ids.clear();
          args.clear();
        }

        depth = r.depth;
        op_idx = r.op_idx;
        pos_idx = r.pos_idx;

        // Collect the arguments for the current call.
        for (tensor_idx_t arg_idx = 0;
             arg_idx < op_input_ts_idx_[r.op_idx].size(); ++arg_idx) {
          args.push_back(
              wiring_results_[std::make_tuple(r.depth, r.op_idx, arg_idx)]
                             [r.pos_idx]);
        }
      }
      result_ids.push_back(i);
    }
    // Add the very last call.
    if (op_idx != -1) {
      CHECK_EQ(result_ids.size(), op_output_ts_idx_[op_idx].size());
#if LOOM_DEDUP_DEBUG
      std::cerr << "Adding op_idx=" << op_idx << " args=" << args
                << " result_ids=" << result_ids << std::endl;
#endif
      cached_calls_.emplace(std::make_pair(op_idx, args), result_ids);
    }
  }

  return true;
}

tensor_idx_t Weaver::Depth(tensor_idx_t result_id) const {
  if (!FastBoundsCheck(result_id, loom_results_.size())) {
    return -1;
  }
  return loom_results_[result_id].depth;
}

tensor_idx_t Weaver::GetTypeShape(tensor_idx_t result_id) const {
  if (!FastBoundsCheck(result_id, loom_results_.size())) {
    return -1;
  }
  return loom_results_[result_id].ts_idx;
}

tensor_idx_t Weaver::GetNamedTensor(
    tensor_idx_t ts_idx, tensor_idx_t named_tensor_idx) {
  if (!FastBoundsCheck(ts_idx, num_type_shapes_)) {
    error_string_ = StrCat("Invalid TypeShape ID: ", ts_idx);
    return -1;
  }
  if (!FastBoundsCheck(named_tensor_idx,
                       num_named_tensors_by_ts_idx_[ts_idx])) {
    error_string_ = StrCat("Invalid NamedTensor ID: ", named_tensor_idx,
                           " for typeshape ", ts_idx, ".");
    return -1;
  }
  if (deduplicate_) {
    auto it =
        cached_named_tensors_.find(std::make_pair(ts_idx, named_tensor_idx));
    if (it != cached_named_tensors_.end()) {
      return it->second;
    }
  }
  tensor_idx_t result_id = loom_results_.size();
  loom_results_.emplace_back();
  LoomResult &r = loom_results_.back();
  r.depth = 0;
  r.ts_idx = ts_idx;
  r.pos_idx = named_tensor_idx;
  r.op_idx = r.op_output_idx = -1;  // unused.
  if (deduplicate_) {
    cached_named_tensors_.emplace(std::make_pair(ts_idx, named_tensor_idx),
                                  result_id);
  }
  return result_id;
}

tensor_idx_t Weaver::MakeConstantSerialized(
    tensor_idx_t ts_idx, const string &tensor_bytes) {
  if (!FastBoundsCheck(ts_idx, num_type_shapes_)) {
    error_string_ = StrCat("Invalid TypeShape ID: ", ts_idx);
    return -1;
  }
  tensor_idx_t dt_size = tensorflow::DataTypeSize(type_shapes_[ts_idx].dtype);
  if (dt_size != 0 && (dt_size * type_shapes_[ts_idx].shape.num_elements() !=
                       tensor_bytes.size())) {
    // Note: we only bother with the size check if dt_size != 0, because when
    // dt_size is zero, that means we're dealing with DataType without a fixed
    // size (like string.)
    error_string_ = StrCat("Invalid serialized tensor passed in; has ",
                           tensor_bytes.size(), " bytes, expected: ",
                           dt_size * type_shapes_[ts_idx].shape.num_elements());
    return -1;
  }

  const auto &ts_metadata = metadata_.type_shape_metadata(ts_idx);
  if (ts_metadata.is_batch_input()) {
    error_string_ = StrCat(
        "Cannot create a constant for a TypeShape ", ts_idx,
        " which is in batch mode.");
    return -1;
  }

  Tensor tensor(type_shapes_[ts_idx].dtype,
                type_shapes_[ts_idx].shape);
  switch (type_shapes_[ts_idx].dtype) {
#define HANDLE_CASE(_tensor_type_) \
    case _tensor_type_: \
      memcpy(tensor.flat<EnumToDataType<_tensor_type_>::Type>().data(), \
             tensor_bytes.data(), tensor_bytes.size()); \
      break;
    HANDLE_CASE(DT_FLOAT);
    HANDLE_CASE(DT_DOUBLE);
    HANDLE_CASE(DT_INT32);
    HANDLE_CASE(DT_UINT16);
    HANDLE_CASE(DT_UINT8);
    HANDLE_CASE(DT_INT16);
    HANDLE_CASE(DT_INT8);
    // HANDLE_CASE(DT_STRING);  // String isn't supported.
    HANDLE_CASE(DT_COMPLEX64);
    HANDLE_CASE(DT_COMPLEX128);
    HANDLE_CASE(DT_INT64);
    HANDLE_CASE(DT_BOOL);
    HANDLE_CASE(DT_QINT8);
    HANDLE_CASE(DT_QUINT8);
    HANDLE_CASE(DT_QINT16);
    HANDLE_CASE(DT_QUINT16);
    HANDLE_CASE(DT_QINT32);
    HANDLE_CASE(DT_BFLOAT16);
    HANDLE_CASE(DT_HALF);
#undef HANDLE_CASE
    default:
      LOG(FATAL) << "Weaver.MakeConstantSerialized does not support tensors "
                 << "of type " << DataType_Name(type_shapes_[ts_idx].dtype);
  }

  return MakeConstant(ts_idx, tensor);
}

tensor_idx_t Weaver::MakeConstant(tensor_idx_t ts_idx,
                           const TensorProto &tensor_proto) {
  Tensor tensor(tensor_proto.dtype());
  if (!tensor.FromProto(tensor_proto)) {
    error_string_ =
        "Converstion from TensorProto to Tensor failed in MakeConstant.";
    return -1;
  }
  return MakeConstant(ts_idx, tensor);
}

tensor_idx_t Weaver::MakeConstant(tensor_idx_t ts_idx, const Tensor &tensor) {
  if (!FastBoundsCheck(ts_idx, num_type_shapes_)) {
    error_string_ = StrCat("Invalid TypeShape ID: ", ts_idx);
    return -1;
  }
  if (tensor.dtype() != type_shapes_[ts_idx].dtype) {
    error_string_ = StrCat(
        "Invalid DType ", tensorflow::DataType_Name(tensor.dtype()),
        " for typeshape ", ts_idx, ".  Expected: ",
        tensorflow::DataType_Name(type_shapes_[ts_idx].dtype));
    return -1;
  }
  if (tensor.shape() != type_shapes_[ts_idx].shape) {
    error_string_ = StrCat(
        "Invalid shape ", tensor.shape().DebugString(),
        " for typeshape ", ts_idx, ".  Expected: ",
        type_shapes_[ts_idx].shape.DebugString());
    return -1;
  }

  if (deduplicate_) {
    auto it = cached_constant_values_.find(std::make_pair(ts_idx, tensor));
    if (it != cached_constant_values_.end()) {
      return it->second.result_id;
    }
  }

  tensor_idx_t constant_value_idx =
      constant_values_by_type_shape_[ts_idx].size();
  constant_values_by_type_shape_[ts_idx].push_back(tensor);
  tensor_idx_t result_id = loom_results_.size();
  loom_results_.emplace_back();
  LoomResult &r = loom_results_.back();
  r.depth = 0;
  r.ts_idx = ts_idx;
  r.pos_idx = num_named_tensors_by_ts_idx_[ts_idx] +
      num_constants_by_type_shape_[ts_idx]++;
  r.op_idx = r.op_output_idx = -1;  // unused.

  if (deduplicate_) {
    CachedConstantValueEntry entry{result_id, constant_value_idx, ts_idx};
    cached_constant_values_.emplace(std::make_pair(ts_idx, tensor),
                                    std::move(entry));
  }

  return result_id;
}

tensor_idx_t Weaver::BatchInput(tensor_idx_t ts_idx, tensor_idx_t batch_idx) {
  const auto &ts_metadata = metadata_.type_shape_metadata(ts_idx);
  if (!ts_metadata.is_batch_input()) {
    error_string_ = StrCat(
        "Cannot create a reference to batch input ", batch_idx,
        " of TypeShape ", ts_idx,
        " because that TypeShape is not in batch mode.");
    return -1;
  }
  if (deduplicate_) {
    auto it =
        cached_batch_input_.find(std::make_pair(ts_idx, batch_idx));
    if (it != cached_batch_input_.end()) {
      return it->second;
    }
  }
  tensor_idx_t result_id = loom_results_.size();
  loom_results_.emplace_back();
  LoomResult &r = loom_results_.back();
  r.depth = 0;
  r.ts_idx = ts_idx;
  r.pos_idx = num_named_tensors_by_ts_idx_[ts_idx] + batch_idx;
  r.op_idx = r.op_output_idx = -1;  // unused.
  if (deduplicate_) {
    cached_batch_input_.emplace(std::make_pair(ts_idx, batch_idx),
                                  result_id);
  }
  return result_id;
}

std::vector<tensor_idx_t> Weaver::CallOp(tensor_idx_t op_idx,
                                  const std::vector<tensor_idx_t> &args) {
  if (!FastBoundsCheck(op_idx, num_ops_)) {
    error_string_ = StrCat("Invalid op ID: ", op_idx);
    return {};
  }
  if (op_input_ts_idx_[op_idx].size() != args.size()) {
    error_string_ = StrCat(
       "Op: ", op_names_[op_idx],
       " Invalid number of arguments:", args.size());
    return {};
  }

  for (tensor_idx_t i = 0; i < args.size(); ++i) {
    tensor_idx_t arg = args[i];
    if (!FastBoundsCheck(arg, loom_results_.size())) {
      error_string_ = StrCat(
          "Op ", op_names_[op_idx], " Arg ", i,
          " was given out of scope ID:", args[i]);
      return {};
    }

    tensor_idx_t expected = op_input_ts_idx_[op_idx][i];
    tensor_idx_t got = loom_results_[arg].ts_idx;
    if (expected != got) {
      error_string_ = StrCat(
          "Op ", op_names_[op_idx], " type mismatch at arg ", i,
          " Expected: ",  type_shapes_[expected].name,
          " got: ", type_shapes_[got].name);
      return {};
    }
  }

  tensor_idx_t max_arg_depth = 0;
  for (tensor_idx_t arg : args) {
    if (loom_results_[arg].depth > max_arg_depth) {
      max_arg_depth = loom_results_[arg].depth;
    }
  }

  if (max_depth_ != -1) {
    if (max_arg_depth >= max_depth_) {
      error_string_ = StrCat("Maximum depth ", max_depth_, " exceeded.");
      return {};
    }
  }

  if (use_tensor_array_) {
    return CallOpInternal(op_idx, args, max_arg_depth);
  }

  std::vector<tensor_idx_t> deepened_args;
  for (tensor_idx_t arg : args) {
    deepened_args.push_back(Deepen(arg, max_arg_depth));
  }

  return CallOpInternal(op_idx, deepened_args, max_arg_depth);
}

std::vector<tensor_idx_t> Weaver::CallOpInternal(
    tensor_idx_t op_idx, const std::vector<tensor_idx_t> &args,
    tensor_idx_t max_arg_depth) {
  tensor_idx_t depth = max_arg_depth + 1;
  if (deepest_ < depth) {
    deepest_ = depth;
  }

  if (deduplicate_) {
    auto it = cached_calls_.find(std::make_pair(op_idx, args));
    if (it != cached_calls_.end()) {
      return it->second;
    }
  }

  tensor_idx_t pos_idx = wiring_results_[
      std::make_tuple(depth, op_idx, 0)].size();
  for (tensor_idx_t i = 0; i < args.size(); ++i) {
    wiring_results_[std::make_tuple(depth, op_idx, i)].push_back(args[i]);
  }

  std::vector<tensor_idx_t> result_ids;
  for (tensor_idx_t i = 0; i < op_output_ts_idx_[op_idx].size(); ++i) {
    tensor_idx_t result_id = loom_results_.size();
    result_ids.push_back(result_id);
    loom_results_.emplace_back();
    LoomResult &r = loom_results_.back();
    r.depth = depth;
    r.ts_idx = op_output_ts_idx_[op_idx][i];
    r.op_idx = op_idx;
    r.op_output_idx = i;
    r.pos_idx = pos_idx;

    if (use_tensor_array_) {
      op_output_wiring_results_[std::make_tuple(depth, op_idx, i)].push_back(
          result_id);
    }
  }

  if (deduplicate_) {
    cached_calls_.emplace(std::make_pair(op_idx, args), result_ids);
  }

  return result_ids;
}

tensor_idx_t Weaver::Deepen(tensor_idx_t result_id, tensor_idx_t target_depth) {
  CHECK(!use_tensor_array_);
  tensor_idx_t ts_idx = loom_results_[result_id].ts_idx;
  tensor_idx_t num_deepenings = target_depth - loom_results_[result_id].depth;
  for (tensor_idx_t i = 0; i < num_deepenings; ++i) {
    if (loom_results_[result_id].cached_passthrough == -1) {
      // Note: This CallOpInternal creates a passthrough because it's relying on
      // the fact that the first `num_type_shapes_` ops are PassThroughs
      // associated with the corresponding TypeShapes.  So `ts_idx` is also the
      // index of the corresponding passthrough op.
      tensor_idx_t new_result_id = CallOpInternal(
          ts_idx, {result_id}, loom_results_[result_id].depth)[0];

      loom_results_[result_id].cached_passthrough = new_result_id;
      result_id = new_result_id;
    } else {
      result_id = loom_results_[result_id].cached_passthrough;
    }
  }
  return result_id;
}

bool Weaver::AddOutput(tensor_idx_t result_id) {
  if (!FastBoundsCheck(result_id, loom_results_.size())) {
    error_string_ = StrCat("AddOutput: result_id ", result_id, " is invalid.");
    return false;
  }
  output_result_ids_.push_back(result_id);
  return true;
}

bool Weaver::MergeFromSerialized(const string &other) {
  WeaverMessage message;
  if (!message.ParseFromString(other)) {
    error_string_ = "WeaverMessage couldn't be parsed.";
    return false;
  }

  if (num_constants_by_type_shape_.size() !=
      message.num_constants_by_type_shape_size()) {
    error_string_ =
        "WeaverMessage didn't have the same number of type-shapes.";
    return false;
  }

  if (deduplicate_) {
    return MergeFromSerializedWithDeduplication(message);
  }

  // The loom_results_ from other will get appended to the current set of
  // loom_results_, so we need to save the size to know how much to offset all
  // the result ID fields from 'other'.
  tensor_idx_t result_id_offset = loom_results_.size();

  // This block copies message's loom results into loom_results_.
  //
  // For LoomResults created by MakeInput, no update need occur if this was one
  // of the inputs shared between the looms (named tensors.)  Otherwise we
  // shift by the number of non-shared inputs in the current Scheduler
  // (constants.)
  //
  // For LoomResults created by CallOp, pos_idx needs to be shifted by the
  // number of times the op has been called at this depth.
  //
  // For any LoomResult, cached_passthrough needs to be shifted by
  // result_id_offset if it was set.
  //
  // Note: this copy needs to happen before wiring results get copied over so
  // that the call-counts are accurate, and before num_inputs_by_type_shape_ is
  // updated.
  tensor_idx_t other_num_loom_results = message.depth_size();
  for (tensor_idx_t i = 0; i < other_num_loom_results; ++i) {
    loom_results_.emplace_back(LoomResultFromMessage(message, i));
    auto &r = loom_results_.back();

    tensor_idx_t pos_idx_offset;
    if (r.depth == 0) {
      if (r.pos_idx < num_named_tensors_by_ts_idx_[r.ts_idx]) {
        // NamedTensor case.
        pos_idx_offset = 0;
      } else {  // Constant/BatchInput case.
        // Note: for BatchInput typeshapes this is always 0.
        pos_idx_offset = num_constants_by_type_shape_[r.ts_idx];
      }
    } else {  // Op output case.
      auto key = std::make_tuple(r.depth, r.op_idx, 0);
      pos_idx_offset = wiring_results_[key].size();
    }
    r.pos_idx += pos_idx_offset;
    if (r.cached_passthrough != -1) {
      r.cached_passthrough += result_id_offset;
    }

    // Update deepest_.
    deepest_ = std::max(deepest_, r.depth);
  }


  // Update num_inputs_by_type_shape_ from message's constants.
  //
  // Also concatenate the lists of constant values.
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    num_constants_by_type_shape_[ts_idx] +=
        message.num_constants_by_type_shape(ts_idx);
    Tensor constants(metadata_.type_shape_metadata(ts_idx).dtype());
    if (!constants.FromProto(message.constant_values_by_type_shape(ts_idx))) {
      error_string_ = StrCat(
          "Conversion from TensorProto to Tensor failed in deserialization.  ",
          "ts_idx=", ts_idx);
      return false;
    }
    auto unstacked = UnstackTensors(constants);
    constant_values_by_type_shape_[ts_idx].insert(
        constant_values_by_type_shape_[ts_idx].end(),
        unstacked.begin(), unstacked.end());
  }

  // Copy over message.wiring into wiring_results_
  // (Shifting all the result IDs by result_id_offset.)
  for (const auto &w : message.wiring()) {
    auto &ids = wiring_results_[
        std::make_tuple(w.depth(), w.op_idx(), w.arg_idx())];
    for (tensor_idx_t result_id : w.result_id()) {
      ids.push_back(result_id_offset + result_id);
    }
  }

  // Copy over message.op_output_wiring into op_output_wiring_results_
  // (Shifting all the result IDs by result_id_offset.)
  for (const auto &w : message.op_output_wiring()) {
    auto &ids = op_output_wiring_results_[
        std::make_tuple(w.depth(), w.op_idx(), w.arg_idx())];
    for (tensor_idx_t result_id : w.result_id()) {
      ids.push_back(result_id_offset + result_id);
    }
  }

  // Copy over outputs_result_ids (shifting by result_id_offset.)
  for (tensor_idx_t result_id : message.output_result_id()) {
    output_result_ids_.push_back(result_id_offset + result_id);
  }
  return true;
}

bool Weaver::MergeFromSerializedWithDeduplication(
    const WeaverMessage &message) {
  // Deserialize all the constants.
  std::vector<std::vector<tensorflow::Tensor>> other_constant_values_by_ts(
      num_type_shapes_);
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    Tensor constants(metadata_.type_shape_metadata(ts_idx).dtype());
    if (!constants.FromProto(message.constant_values_by_type_shape(ts_idx))) {
      error_string_ = StrCat(
          "Conversion from TensorProto to Tensor failed in deserialization.  ",
          "ts_idx=", ts_idx);
      return false;
    }
    other_constant_values_by_ts[ts_idx] = UnstackTensors(constants);
  }

  // Deserialize LoomResults from the other message.
  std::vector<LoomResult> other_results;
  for (tensor_idx_t i = 0; i < message.depth_size(); ++i) {
    other_results.emplace_back(LoomResultFromMessage(message, i));
    // Update deepest_.
    deepest_ = std::max(deepest_, other_results.back().depth);
  }
  std::vector<std::vector<tensor_idx_t>> other_results_by_depth(1);
  for (tensor_idx_t i = 0; i < other_results.size(); ++i) {
    tensor_idx_t depth = other_results[i].depth;
    enlarge(&other_results_by_depth, depth);
    other_results_by_depth[depth].push_back(i);
  }

  std::vector<tensor_idx_t> result_id_map(other_results.size(), -1);

  // Key: depth, op_idx, pos_idx
  // Value: args
  std::unordered_map<std::tuple<tensor_idx_t, tensor_idx_t, tensor_idx_t>,
                     std::vector<tensor_idx_t>>
      other_calls;
  for (const auto &w : message.wiring()) {
    for (tensor_idx_t pos_idx = 0; pos_idx < w.result_id_size(); ++pos_idx) {
      auto& args = other_calls[std::make_tuple(w.depth(), w.op_idx(), pos_idx)];
      enlarge(&args, w.arg_idx());
      args[w.arg_idx()] = w.result_id(pos_idx);
    }
  }

  // Go through all LoomResults by depth.
  // Depth 0:
  // - Batch input
  // - Named tensors
  // - Constants
  // Depth >0: CallOps
  //
  // For each LoomResult, allocate an ID:
  // - An existing ID, if there's a suitable one
  //   - Batch input: cached_batch_input_
  //   - Named tensors: cached_named_tensors_
  //   - Constants: cached_constant_values_
  //   - CallOp: cached_calls_
  // - A new ID, by appending to loom_result_.

  for (tensor_idx_t id : other_results_by_depth[0]) {
    LoomResult& r = other_results[id];
    if (r.pos_idx < num_named_tensors_by_ts_idx_[r.ts_idx]) {
      // Named tensor.
      const auto &key = std::make_pair(r.ts_idx, r.pos_idx);
      auto it = cached_named_tensors_.find(key);
      if (it == cached_named_tensors_.end()) {
        result_id_map[id] = loom_results_.size();
        cached_named_tensors_.emplace(key, loom_results_.size());
        loom_results_.push_back(r);
      } else {
        result_id_map[id] = it->second;
      }
    } else if (metadata_.type_shape_metadata(r.ts_idx).is_batch_input()) {
      // Batch input.
      const auto &key = std::make_pair(
          r.ts_idx, r.pos_idx - num_named_tensors_by_ts_idx_[r.ts_idx]);
      auto it = cached_batch_input_.find(key);
      if (it == cached_batch_input_.end()) {
        result_id_map[id] = loom_results_.size();
        cached_batch_input_.emplace(key, loom_results_.size());
        loom_results_.push_back(r);
      } else {
        result_id_map[id] = it->second;
      }
    } else {
      // Constant.
      const tensorflow::Tensor &tensor = other_constant_values_by_ts
          [r.ts_idx][r.pos_idx - num_named_tensors_by_ts_idx_[r.ts_idx]];
#if LOOM_DEDUP_DEBUG
      std::cerr << "Processing constant LR with ID " << id << std::endl;
      std::cerr << "Value: " << tensor.DebugString() << std::endl;
#endif
      const auto& key = std::make_pair(r.ts_idx, tensor);
      auto it = cached_constant_values_.find(key);
      if (it == cached_constant_values_.end()) {
#if LOOM_DEDUP_DEBUG
        std::cerr << "> Not found, new ID " << loom_results_.size()
                  << std::endl;
#endif
        r.pos_idx = constant_values_by_type_shape_[r.ts_idx].size();
        result_id_map[id] = loom_results_.size();
        CachedConstantValueEntry ent{loom_results_.size(), r.pos_idx, r.ts_idx};
        cached_constant_values_.emplace(key, std::move(ent));
        loom_results_.push_back(r);
        constant_values_by_type_shape_[r.ts_idx].push_back(std::move(tensor));
        num_constants_by_type_shape_[r.ts_idx]++;
      } else {
#if LOOM_DEDUP_DEBUG
        std::cerr << "> Found, existing ID " << it->second.result_id
                  << std::endl;
#endif
        result_id_map[id] = it->second.result_id;
      }
    }
  }

  tensor_idx_t depth = -1, op_idx = -1, pos_idx = -1;
  std::vector<tensor_idx_t> result_ids;
  auto process_results = [&]() {
    const auto& other_args =
        other_calls[std::make_tuple(depth, op_idx, pos_idx)];
    std::vector<tensor_idx_t> args;
    for (tensor_idx_t arg : other_args) {
      args.push_back(result_id_map[arg]);
      CHECK_NE(args.back(), -1);
    }

    const auto &key = std::make_pair(op_idx, args);
#if LOOM_DEDUP_DEBUG
    std::cerr << "Looking for call to " << op_idx << " with args " << args
              << std::endl;
#endif
    auto it = cached_calls_.find(key);
    if (it == cached_calls_.end()) {
      // Make new LoomResults.
      tensor_idx_t new_pos_idx =
          wiring_results_[std::make_tuple(depth, op_idx, 0)].size();
      std::vector<tensor_idx_t> new_result_ids;
      for (tensor_idx_t i = 0; i < result_ids.size(); ++i) {
        LoomResult &r = other_results[result_ids[i]];
        result_id_map[result_ids[i]] = loom_results_.size();
        new_result_ids.push_back(loom_results_.size());
        r.pos_idx = new_pos_idx;
        loom_results_.push_back(r);
      }
      for (tensor_idx_t i = 0; i < args.size(); ++i) {
        wiring_results_[std::make_tuple(depth, op_idx, i)].push_back(args[i]);
      }
      cached_calls_.emplace(key, new_result_ids);
#if LOOM_DEDUP_DEBUG
      std::cerr << "> Not found, creating results " << new_result_ids << std::endl;
#endif
    } else {
#if LOOM_DEDUP_DEBUG
      std::cerr << "> Found, existing";
#endif
      for (tensor_idx_t i = 0; i < result_ids.size(); ++i) {
        result_id_map[result_ids[i]] = it->second[i];
#if LOOM_DEDUP_DEBUG
        std:: cerr << ' ' << it->second[i];
#endif
      }
#if LOOM_DEDUP_DEBUG
      std::cerr << std::endl;
#endif
    }
    result_ids.clear();
  };

  for (depth = 1; depth < other_results_by_depth.size(); ++depth) {
    for (tensor_idx_t id : other_results_by_depth[depth]) {
      const LoomResult& lr = other_results[id];
#if LOOM_DEDUP_DEBUG
      std::cerr << "LR id=" << id << " depth=" << lr.depth
                << " ts_idx=" << lr.ts_idx << " op_idx=" << lr.op_idx
                << " op_output_idx=" << lr.op_output_idx
                << " pos_idx=" << lr.pos_idx << std::endl;
      std::cerr << "op_idx=" << op_idx << " pos_idx=" << pos_idx << std::endl;
#endif
      if (op_idx != other_results[id].op_idx ||
          pos_idx != other_results[id].pos_idx) {
        if (!result_ids.empty()) {
          process_results();
        }
        op_idx = other_results[id].op_idx;
        pos_idx = other_results[id].pos_idx;
      }
      result_ids.push_back(id);
    }
    if (op_idx != -1) {
      process_results();
    }
  }

  // Copy over outputs_result_ids (shifting by result_id_offset.)
  for (tensor_idx_t result_id : message.output_result_id()) {
    output_result_ids_.push_back(result_id_map[result_id]);
  }
  return true;

#if 0
  // See if any of the constants are the same.
  // First, get all LoomResults which refer to constants.
  //
  // (ts_idx, pos_idx) -> LoomResult ID
  // It might be fine to make this a map<ts_idx, vector<result_id>>
  // if pos_idx and result_id are always in the same order.
  std::unordered_map<std::pair<tensor_idx_t, tensor_idx_t>, tensor_idx_t>
      other_constant_loom_result_ids;
  for (tensor_idx_t i = 0; i < other_results.size(); ++i) {
    const LoomResult& lr = std::get<0>(other_results[i]);
    if (!metadata_.type_shape_metadata(lr.ts_idx).is_batch_input() &&
        lr.depth == 0 &&
        lr.pos_idx > num_named_tensors_by_ts_idx_[lr.ts_idx]) {
      other_constant_loom_result_ids.emplace(
          std::make_pair(
              lr.ts_idx,
              lr.pos_idx - num_named_tensors_by_ts_idx_[lr.ts_idx]),
          i);
    }
  }

  // Now, for each constant, see if it's the same as an existing constant.
  // If so, remap all result IDs to be the same.
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    Tensor constants(metadata_.type_shape_metadata(ts_idx).dtype());
    if (!constants.FromProto(message.constant_values_by_type_shape(ts_idx))) {
      error_string_ = StrCat(
          "Conversion from TensorProto to Tensor failed in deserialization.  ",
          "ts_idx=", ts_idx);
      return false;
    }

    auto unstacked = UnstackTensors(constants);
    for (int i = 0; i < unstacked.size(); ++i) {
      const auto other_id =
          other_constant_loom_results_ids[std::make_pair(ts_idx, i)];
      const auto& entry = other_results[other_id];
      const auto& tensor = unstacked[i];

      auto it = cached_constant_values_.find(std::make_pair(ts_idx, tensor));
      if (it == cached_constant_values_.end()) {
        constant_values_by_type_shape_[ts_idx].emplace_back(std::move(tensor));
        std::get<0>(entry).pos_idx = num_constants_by_type_shape_[ts_idx]++;
      } else {
        // This is the new result ID number
        std::get<1>(entry) = *it;
        // Yes, this has been remapped
        std::get<2>(entry) = true;
      }
    }
  }

  // Go through all LoomResults created through CallOp.
  // For each LoomResult, we can find out:
  // - op name
  // - argument LoomResult numbers
  //
  // 1. Use CallOpInternal.
  // Remap all argument LoomResults to their new numbers.
  // Use CallOpInternal with the new argument numbers.
  // Map the original LoomResultID to resulting result ID
#endif
}

void Weaver::Finalize() {
  // Make sure Finalize only does anything the first time.
  if (finalized_) return;
  finalized_ = true;

  if (max_depth_ == -1) {
    max_depth_ = Deepest();
  }

  if (use_tensor_array_) {
    FinalizeTensorArray();
  } else {
    FinalizeConventional();
  }

}

void Weaver::FinalizeTensorArray() {
  std::vector<tensor_idx_t> result_id_to_type_shape_ta_idx(loom_results_.size(),
                                                           -1);
  num_results_per_type_shape_.resize(num_type_shapes_);

  // Map each LoomResult from an op to an index for the TensorArray of that
  // TypeShape.  Do this so that the outputs of an op are contiguous.
  for (tensor_idx_t depth = 1; depth <= max_depth_; ++depth) {
    for (tensor_idx_t op_idx = 0; op_idx < num_ops_; ++op_idx) {
      for (tensor_idx_t output_idx = 0;
           output_idx < op_output_ts_idx_[op_idx].size(); ++output_idx) {
        tensor_idx_t ts_idx = op_output_ts_idx_[op_idx][output_idx];
        for (tensor_idx_t id : op_output_wiring_results_[std::make_tuple(
                 depth, op_idx, output_idx)]) {
          result_id_to_type_shape_ta_idx[id] =
              num_results_per_type_shape_[ts_idx]++;
        }
      }
    }
  }

  // Map LoomResults for constants, batch inputs, and constants.
  // We don't know in advance how many batch inputs there will be, so we put
  // them in a contiguous block at the end of the TensorArray.
  for (tensor_idx_t i = 0; i < loom_results_.size(); ++i) {
    const auto &loom_result = loom_results_[i];
    if (loom_result.depth != 0) {
      continue;
    }
    result_id_to_type_shape_ta_idx[i] =
        num_results_per_type_shape_[loom_result.ts_idx] + loom_result.pos_idx;
  }

  // Populate 'final_wiring_' and final_op_output_wiring_ with the values that
  // will end up in the wiring diagram placeholder variables in the loom.
  for (tensor_idx_t depth = 1; depth <= max_depth_; ++depth) {
    for (tensor_idx_t op_idx = 0; op_idx < num_ops_; ++op_idx) {
      for (tensor_idx_t arg_idx = 0; arg_idx < op_input_ts_idx_[op_idx].size();
           ++arg_idx) {
        auto key = std::make_tuple(depth, op_idx, arg_idx);
        const std::vector<tensor_idx_t> &my_wiring_results =
            wiring_results_[key];
        std::vector<tensor_idx_t> &my_final_wiring = final_wiring_[key];
        for (tensor_idx_t result_id : my_wiring_results) {
          my_final_wiring.push_back(result_id_to_type_shape_ta_idx[result_id]);
        }
      }
    }
  }

  // Populate 'final_output_wiring_'
  final_output_wiring_.resize(num_type_shapes_);
  for (tensor_idx_t output_result_id : output_result_ids_) {
    const LoomResult &r = loom_results_[output_result_id];
    final_output_wiring_[r.ts_idx].push_back(
        result_id_to_type_shape_ta_idx[output_result_id]);
  }
}

void Weaver::FinalizeConventional() {
  // Add PassThroughs so that all the outputs are at the 'max_depth_'.
  std::vector<tensor_idx_t> deepened_outputs;
  for (tensor_idx_t output_result_id : output_result_ids_) {
    deepened_outputs.push_back(Deepen(output_result_id, max_depth_));
  }

  // Given a depth, an op_idx and an op_output_idx, wiring_offset tells us where
  // in the state Tensor of the appropriate type_shape the 'op_output_idx'th
  // output of 'op' will start, after the concat which takes place in sub-layer
  // (3) of the previous loom layer.
  std::map<std::tuple<tensor_idx_t, tensor_idx_t, tensor_idx_t>,
           tensor_idx_t> wiring_offset;

  // Named tensors, constants and batch input members have depth=0, op_idx=-1
  // and op_output_idx=-1, and we don't have to add anything to their 'pos_idx'
  // to get their index in the initial state tensor.
  wiring_offset[std::make_tuple(0, -1, -1)] = 0;

  for (tensor_idx_t depth = 1; depth <= max_depth_; ++depth) {
    std::vector<tensor_idx_t> concat_offset(num_type_shapes_);
    for (tensor_idx_t op_idx = 0; op_idx < num_ops_; ++op_idx) {
      // Note: the number of times op 'op_idx' got called at 'depth' is the same
      // as the number of 0th arguments passed in.
      tensor_idx_t num_op_calls = wiring_results_[
            std::make_tuple(depth, op_idx, 0)].size();

      for (tensor_idx_t output_idx = 0;
           output_idx < op_output_ts_idx_[op_idx].size();
           ++output_idx) {
        // The TypeShape for the current output determines which state tensor
        // this batch of outputs ended up in.
        tensor_idx_t ts_idx = op_output_ts_idx_[op_idx][output_idx];

        wiring_offset[std::make_tuple(depth, op_idx, output_idx)] =
            concat_offset[ts_idx];

        // We're incrementing concat_offset by 'num_op_calls' because the
        // current op must have 'num_op_calls' as the batch size for all of its
        // outputs (including this one.)
        concat_offset[ts_idx] += num_op_calls;
      }
    }
  }

  // Populate 'final_wiring_' with the values that will end up in the wiring
  // diagram placeholder variables in the loom.
  for (tensor_idx_t depth = 1; depth <= max_depth_; ++depth) {
    for (tensor_idx_t op_idx = 0; op_idx < num_ops_; ++op_idx) {
      for (tensor_idx_t arg_idx = 0; arg_idx < op_input_ts_idx_[op_idx].size();
           ++arg_idx) {
        auto key = std::make_tuple(depth, op_idx, arg_idx);
        const std::vector<tensor_idx_t> &my_wiring_results =
            wiring_results_[key];
        std::vector<tensor_idx_t> &my_final_wiring = final_wiring_[key];
        for (tensor_idx_t result_id : my_wiring_results) {
          const LoomResult &r = loom_results_[result_id];
          my_final_wiring.push_back(
              r.pos_idx + wiring_offset[std::make_tuple(
                  r.depth, r.op_idx, r.op_output_idx)]);
        }
      }
    }
  }

  // Populate 'final_output_wiring_'
  final_output_wiring_.resize(num_type_shapes_);
  for (tensor_idx_t deepened_output : deepened_outputs) {
    const LoomResult &r = loom_results_[deepened_output];
    final_output_wiring_[r.ts_idx].push_back(
        r.pos_idx + wiring_offset[std::make_tuple(
            r.depth, r.op_idx, r.op_output_idx)]);
  }
}

}  // namespace fold
}  // namespace tensorflow

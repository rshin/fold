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

#include <iostream>
#include <stdio.h>

#include <gtest/gtest.h>

#include "tensorflow_fold/loom/loom.pb.h"
#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace fold {

namespace {

Tensor FloatVectorTensor(const std::vector<float> &vec) {
  TensorShape shape;
  shape.AddDim(vec.size());
  Tensor tensor(DT_FLOAT, shape);
  auto flat = tensor.template flat<float>();
  for (size_t i = 0; i < vec.size(); ++i) {
    flat(i) = vec[i];
  }
  return tensor;
}

LoomMetadata MakeLoomMetadata() {
  LoomMetadata metadata;
  metadata.set_max_depth(-1);

  // TypeShape #0 : float vector of length 3.
  auto *type_shape_metadata = metadata.add_type_shape_metadata();
  type_shape_metadata->set_dtype(DT_FLOAT);
  type_shape_metadata->add_shape(3);
  type_shape_metadata->set_name("3vec");
  type_shape_metadata->set_is_batch_input(false);

  // Op #0 : mandatory pass through op for TypeShape #0
  auto *passthrough0_op_metadata = metadata.add_op_metadata();
  passthrough0_op_metadata->set_name("pass-through 0");
  passthrough0_op_metadata->add_input_ts_idx(0);
  passthrough0_op_metadata->add_output_ts_idx(0);

  // Op #1: binary op 'plus'.
  auto *plus_op_metadata = metadata.add_op_metadata();
  plus_op_metadata->set_name("plus");
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_output_ts_idx(0);

  return metadata;
}

}  // namespace

TEST(WeaverTest, BuildMetadata) {
  LoomMetadata metadata = MakeLoomMetadata();

  string error_string;
  CHECK(VerifyLoomMetadata(metadata, &error_string)) << error_string;

  string metadata_str;
  metadata.SerializeToString(&metadata_str);
  Weaver w(metadata_str);

  std::vector<tensor_idx_t> constant_results;
  for (size_t i = 0; i < 6; ++i) {
    constant_results.push_back(w.MakeConstant(0, FloatVectorTensor(
        {1.0f * i, 2.0f * i, 3.0f * i})));
  }

  std::vector<tensor_idx_t> pair_results;
  for (size_t i = 0; i < 3; ++i) {
    pair_results.push_back(w.CallOp(1, {
      constant_results[2 * i], constant_results[2 * i + 1]})[0]);
  }

  tensor_idx_t first_four_result = w.CallOp(1, {
    pair_results[0], pair_results[1]})[0];
  tensor_idx_t total_result = w.CallOp(
     1, {first_four_result, pair_results[2]})[0];
  w.AddOutput(total_result);
  w.Finalize();

  Tensor constants_tensor = w.BatchConstantValues(0);
  auto constants = constants_tensor.flat_inner_dims<float>();
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(constants(i, j),  i * (j + 1));
    }
  }

  // Level 1 is where the pair_results get constructed.
  EXPECT_TRUE(w.GetWiring(1, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level1_plus_arg0 = w.GetWiring(1, 1, 0);
  const std::vector<tensor_idx_t> &level1_plus_arg1 = w.GetWiring(1, 1, 1);

  EXPECT_EQ(3, level1_plus_arg0.size());
  EXPECT_EQ(3, level1_plus_arg1.size());
  EXPECT_EQ(0, level1_plus_arg0[0]);
  EXPECT_EQ(1, level1_plus_arg1[0]);
  EXPECT_EQ(2, level1_plus_arg0[1]);
  EXPECT_EQ(3, level1_plus_arg1[1]);
  EXPECT_EQ(4, level1_plus_arg0[2]);
  EXPECT_EQ(5, level1_plus_arg1[2]);

  // Level 2 has one call to plus (for pair_results[0] and pair_results[1]), and
  // a pass through (for pair_results[2])
  const std::vector<tensor_idx_t> &level2_passthrough_arg0 = w.GetWiring(
      2, 0, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg0 = w.GetWiring(2, 1, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg1 = w.GetWiring(2, 1, 1);
  EXPECT_EQ(1, level2_passthrough_arg0.size());
  EXPECT_EQ(1, level2_plus_arg0.size());
  EXPECT_EQ(1, level2_plus_arg1.size());
  EXPECT_EQ(0, level2_plus_arg0[0]);
  EXPECT_EQ(1, level2_plus_arg1[0]);
  EXPECT_EQ(2, level2_passthrough_arg0[0]);

  // Level 3 has one call to plus.  Its arguments are in reverse order because
  // the pass-through has a lower op-index than plus, so the result from
  // pair_results[2] ends up in position 0, while first_four_result ends up in
  // position 1.
  EXPECT_TRUE(w.GetWiring(3, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level3_plus_arg0 = w.GetWiring(3, 1, 0);
  const std::vector<tensor_idx_t> &level3_plus_arg1 = w.GetWiring(3, 1, 1);
  EXPECT_EQ(1, level3_plus_arg0.size());
  EXPECT_EQ(1, level3_plus_arg1.size());
  EXPECT_EQ(1, level3_plus_arg0[0]);
  EXPECT_EQ(0, level3_plus_arg1[0]);

  // Output Wiring: straight forward as only one output has been marked and only
  // one value exists at level 3.
  const std::vector<tensor_idx_t> &output_wiring = w.GetOutputWiring(0);
  EXPECT_EQ(1, output_wiring.size());
  EXPECT_EQ(0, output_wiring[0]);
}

TEST(WeaverTest, MergeWithDeduplication) {
  LoomMetadata metadata = MakeLoomMetadata();
  metadata.set_deduplicate(true);

  string error_string;
  CHECK(VerifyLoomMetadata(metadata, &error_string)) << error_string;

  string metadata_str;
  metadata.SerializeToString(&metadata_str);

  // Setup Weaver for w1
  Weaver w1(metadata_str);
  std::vector<tensor_idx_t> constant_results;
  for (size_t i = 0; i < 6; ++i) {
    // 0, 1, 2, 3, 4, 5
    constant_results.push_back(w1.MakeConstant(0, FloatVectorTensor(
        {1.0f * i, 2.0f * i, 3.0f * i})));
  }
  std::vector<tensor_idx_t> pair_results;
  for (size_t i = 0; i < 3; ++i) {
    // 0 + 1 (=6), 2 + 3 (=7), 4 + 5 (=8)
    pair_results.push_back(w1.CallOp(1, {
      constant_results[2 * i], constant_results[2 * i + 1]})[0]);
  }
  // (0 + 1) + (2 + 3) (=9)
  tensor_idx_t first_four_result = w1.CallOp(1, {
    pair_results[0], pair_results[1]})[0];
  // ((0 + 1) + (2 + 3)) + (4 + 5)
  tensor_idx_t total_result = w1.CallOp(
      1, {first_four_result, pair_results[2]})[0];
  w1.AddOutput(total_result);

  // Setup Weaver for w2
  Weaver w2(metadata_str);
  constant_results.clear();
  for (size_t i = 2; i < 8; ++i) {
    // 2, 3, 4, 5, 6 (= 13), 7 (= 14)
    constant_results.push_back(w2.MakeConstant(0, FloatVectorTensor(
        {1.0f * i, 2.0f * i, 3.0f * i})));
  }
  pair_results.clear();
  for (size_t i = 0; i < 3; ++i) {
    // 2 + 3, 4 + 5, 6 + 7 -> 15
    pair_results.push_back(w2.CallOp(1, {
      constant_results[2 * i], constant_results[2 * i + 1]})[0]);
  }
  // (2 + 3) + (4 + 5) -> 16
  first_four_result = w2.CallOp(1, {
    pair_results[0], pair_results[1]})[0];
  // ((2 + 3) + (4 + 5)) + (6 + 7)
  total_result = w2.CallOp(
      1, {first_four_result, pair_results[2]})[0];
  // Force pair_results[1] (4 + 5) to be sent through passthrough
  // ((2 + 3) + (4 + 5)) + (4 + 5)
  w2.CallOp(1, {first_four_result, pair_results[1]});
  w2.AddOutput(total_result);

  // Merge twice, should be idempotent
  w1.MergeFromSerialized(w2.Serialize());
  w1.MergeFromSerialized(w2.Serialize());

  // Deserialize and then serialize again
  Weaver w(metadata_str);
  w.Deserialize(w1.Serialize());
  w.MergeFromSerialized(w2.Serialize());
  w.Finalize();

  // Level 1
  // We should have 0 + 1, 2 + 3, 4 + 5, 6 + 7
  EXPECT_TRUE(w.GetWiring(1, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level1_plus_arg0 = w.GetWiring(1, 1, 0);
  const std::vector<tensor_idx_t> &level1_plus_arg1 = w.GetWiring(1, 1, 1);

  EXPECT_EQ(4, level1_plus_arg0.size());
  EXPECT_EQ(4, level1_plus_arg1.size());
  EXPECT_EQ(0, level1_plus_arg0[0]);
  EXPECT_EQ(1, level1_plus_arg1[0]);
  EXPECT_EQ(2, level1_plus_arg0[1]);
  EXPECT_EQ(3, level1_plus_arg1[1]);
  EXPECT_EQ(4, level1_plus_arg0[2]);
  EXPECT_EQ(5, level1_plus_arg1[2]);
  EXPECT_EQ(6, level1_plus_arg0[3]);
  EXPECT_EQ(7, level1_plus_arg1[3]);

  // Level 2 has two calls to plus
  //   (0 + 1) + (2 + 3), (2 + 3) + (4 + 5)
  // and two passthroughs: (4 + 5) and (6 + 7)
  const std::vector<tensor_idx_t> &level2_passthrough_arg0 = w.GetWiring(
      2, 0, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg0 = w.GetWiring(2, 1, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg1 = w.GetWiring(2, 1, 1);
  EXPECT_EQ(2, level2_passthrough_arg0.size());
  EXPECT_EQ(2, level2_plus_arg0.size());
  EXPECT_EQ(2, level2_plus_arg1.size());
  EXPECT_EQ(0, level2_plus_arg0[0]);
  EXPECT_EQ(1, level2_plus_arg1[0]);
  EXPECT_EQ(1, level2_plus_arg0[1]);
  EXPECT_EQ(2, level2_plus_arg1[1]);
  EXPECT_EQ(2, level2_passthrough_arg0[0]);
  EXPECT_EQ(3, level2_passthrough_arg0[1]);

  // Level 3 has three calls to plus.
  //   ((0 + 1) + (2 + 3)) + (4 + 5)
  //   ((2 + 3) + (4 + 5)) + (6 + 7)
  //   ((2 + 3) + (4 + 5)) + (4 + 5)
  // (4 + 5) and (6 + 7) come from passthroughs, and get pos 0 and 1
  // respectively.
  // ((0 + 1) + (2 + 3)) and ((2 + 3) + (4 + 5)) are assigned 2 and 3.
  EXPECT_TRUE(w.GetWiring(3, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level3_plus_arg0 = w.GetWiring(3, 1, 0);
  const std::vector<tensor_idx_t> &level3_plus_arg1 = w.GetWiring(3, 1, 1);
  EXPECT_EQ(3, level3_plus_arg0.size());
  EXPECT_EQ(3, level3_plus_arg1.size());
  EXPECT_EQ(2, level3_plus_arg0[0]);
  EXPECT_EQ(0, level3_plus_arg1[0]);
  EXPECT_EQ(3, level3_plus_arg0[1]);
  EXPECT_EQ(1, level3_plus_arg1[1]);
  EXPECT_EQ(3, level3_plus_arg0[2]);
  EXPECT_EQ(0, level3_plus_arg1[2]);
}

TEST(WeaverTest, BuildMetadataWithTensorArrays) {
  LoomMetadata metadata;
  metadata.set_max_depth(-1);
  metadata.set_use_tensor_array(true);

  // TypeShape #0 : float vector of length 3.
  auto *type_shape_metadata = metadata.add_type_shape_metadata();
  type_shape_metadata->set_dtype(DT_FLOAT);
  type_shape_metadata->add_shape(3);
  type_shape_metadata->set_name("3vec");
  type_shape_metadata->set_is_batch_input(false);

  // TypeShape #1 : float vector of length 4.
  type_shape_metadata = metadata.add_type_shape_metadata();
  type_shape_metadata->set_dtype(DT_FLOAT);
  type_shape_metadata->add_shape(4);
  type_shape_metadata->set_name("4vec");
  type_shape_metadata->set_is_batch_input(false);

  // Op #0: binary op 'plus3'.
  auto *plus_op_metadata = metadata.add_op_metadata();
  plus_op_metadata->set_name("plus3");
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_output_ts_idx(0);

  // Op #1: binary op 'plus4'.
  plus_op_metadata = metadata.add_op_metadata();
  plus_op_metadata->set_name("plus4");
  plus_op_metadata->add_input_ts_idx(1);
  plus_op_metadata->add_input_ts_idx(1);
  plus_op_metadata->add_output_ts_idx(1);

  string error_string;
  CHECK(VerifyLoomMetadata(metadata, &error_string)) << error_string;

  string metadata_str;
  metadata.SerializeToString(&metadata_str);
  Weaver w(metadata_str);

  std::vector<tensor_idx_t> constant_results;
  for (size_t i = 0; i < 6; ++i) {
    constant_results.push_back(w.MakeConstant(0, FloatVectorTensor(
        {1.0f * i, 2.0f * i, 3.0f * i})));
  }

  std::vector<tensor_idx_t> pair_results;
  for (size_t i = 0; i < 3; ++i) {
    pair_results.push_back(w.CallOp(0, {
      constant_results[2 * i], constant_results[2 * i + 1]})[0]);
  }

  tensor_idx_t first_four_result = w.CallOp(0, {
    pair_results[0], pair_results[1]})[0];
  tensor_idx_t total_result = w.CallOp(
      0, {first_four_result, pair_results[2]})[0];
  w.AddOutput(total_result);
  w.Finalize();

  Tensor constants_tensor = w.BatchConstantValues(0);
  auto constants = constants_tensor.flat_inner_dims<float>();
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(constants(i, j),  i * (j + 1));
    }
  }

  auto num_results_per_ts = w.GetNumResultsPerTypeShape();
  EXPECT_EQ(2, num_results_per_ts.size());
  EXPECT_EQ(5, num_results_per_ts[0]);
  EXPECT_EQ(0, num_results_per_ts[1]);

  // Level 1 is where the pair_results get constructed.
  const std::vector<tensor_idx_t> &level1_plus_arg0 = w.GetWiring(1, 0, 0);
  const std::vector<tensor_idx_t> &level1_plus_arg1 = w.GetWiring(1, 0, 1);

  EXPECT_EQ(3, level1_plus_arg0.size());
  EXPECT_EQ(3, level1_plus_arg1.size());
  EXPECT_EQ(5, level1_plus_arg0[0]);
  EXPECT_EQ(6, level1_plus_arg1[0]);
  EXPECT_EQ(7, level1_plus_arg0[1]);
  EXPECT_EQ(8, level1_plus_arg1[1]);
  EXPECT_EQ(9, level1_plus_arg0[2]);
  EXPECT_EQ(10, level1_plus_arg1[2]);

  EXPECT_TRUE(w.GetWiring(1, 1, 0).empty());  // No calls to plus4.

  // Level 2 has one call to plus (for pair_results[0] and pair_results[1]), and
  // a pass through (for pair_results[2])
  const std::vector<tensor_idx_t> &level2_plus_arg0 = w.GetWiring(2, 0, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg1 = w.GetWiring(2, 0, 1);
  EXPECT_EQ(1, level2_plus_arg0.size());
  EXPECT_EQ(1, level2_plus_arg1.size());
  EXPECT_EQ(0, level2_plus_arg0[0]);
  EXPECT_EQ(1, level2_plus_arg1[0]);

  EXPECT_TRUE(w.GetWiring(2, 1, 0).empty());  // No calls to plus4.

  // Level 3 has one call to plus.  Its arguments are in reverse order because
  // the pass-through has a lower op-index than plus, so the result from
  // pair_results[2] ends up in position 0, while first_four_result ends up in
  // position 1.
  const std::vector<tensor_idx_t> &level3_plus_arg0 = w.GetWiring(3, 0, 0);
  const std::vector<tensor_idx_t> &level3_plus_arg1 = w.GetWiring(3, 0, 1);
  EXPECT_EQ(1, level3_plus_arg0.size());
  EXPECT_EQ(1, level3_plus_arg1.size());
  EXPECT_EQ(3, level3_plus_arg0[0]);
  EXPECT_EQ(2, level3_plus_arg1[0]);

  EXPECT_TRUE(w.GetWiring(3, 1, 0).empty());  // No calls to plus4.

  // Output Wiring: straight forward as only one output has been marked and only
  // one value exists at level 3.
  const std::vector<tensor_idx_t> &output_wiring = w.GetOutputWiring(0);
  EXPECT_EQ(1, output_wiring.size());
  EXPECT_EQ(4, output_wiring[0]);
}

}  // namespace fold
}  // namespace tensorflow

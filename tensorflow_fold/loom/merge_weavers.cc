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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow_fold/loom/weaver_op_base.h"

namespace tensorflow {
namespace fold {

REGISTER_OP("MergeWeavers")
    .Attr("metadata: string")
    .Input("weaver_messages: string")
    .Output("merged: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->Scalar());
    });

class MergeWeaversOp : public tensorflow::OpKernel {
 public:
  explicit MergeWeaversOp(tensorflow::OpKernelConstruction *c) {
    OP_REQUIRES_OK(c, c->GetAttr("metadata", &metadata_str_));
  }

  tensorflow::Status Compute(tensorflow::OpKernelContext *c) override {
    Weaver weaver(metadata_str_);
    OP_REQUIRES(c, weaver.error_string().empty(), InvalidArgument(
        "Couldn't initialize weaver from metadata: ", weaver.error_string()));

    auto weaver_messages = c->input(0).flat<string>();

    if (weaver_messages.size() < 1) {
      return tensorflow::errors::InvalidArgument(
          "weaver_messages must contain at least one value.");
    }
    if (!weaver->Deserialize(weaver_messages(0))) {
      return tensorflow::errors::Internal(
          "Failed to deserialize WeaverMessage: ", weaver->error_string());
    }

    // Note: If necessary, this loop could be sped up by merging the messages in
    // a multi-threaded way instead of in sequence.
    // With deduplication, using a tree structure might not really help so much.
    for (int64 i = 1; i < weaver_messages.size(); ++i) {
      if (!weaver->MergeFromSerialized(weaver_messages(i))) {
        return tensorflow::errors::Internal("Failed to merge WeaverMessage", i,
                                            ":", weaver->error_string());
      }
    }

    Tensor* output;
    OP_REQUIRES_OK(c,
                   c->allocate_output(0, tensorflow::TensorShape(), &output));
    output->flat<string>()(0) = weaver.Serialize();

    return tensorflow::Status::OK();
  }

 private:
  string metadata_str_;
  LoomMetadata metadata_;
};

REGISTER_KERNEL_BUILDER(
    Name("MergeWeavers").Device(tensorflow::DEVICE_CPU),
    MergeWeaverOp);

}  // namespace fold
}  // namespace tensorflow

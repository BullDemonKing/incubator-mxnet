/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"
#include "./mkldnn/mkldnn_ops-inl.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

static bool ConcatShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape) {
  using namespace mshadow;
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  TShape dshape;
  index_t size = 0;
  bool has_zero = false;
  int axis = -1;
  for (int i = 0; i < param_.num_args; ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      axis = CheckAxis(param_.dim, tmp.ndim());
      has_zero = tmp[axis] == 0 || has_zero;
      size += tmp[axis];
      tmp[axis] = 0;
      shape_assign(&dshape, tmp);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    axis = CheckAxis(param_.dim, tmp.ndim());
    tmp[axis] = 0;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == 0) return false;

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_zero) dshape[axis] = size;
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return dshape.Size() != 0;
}

static bool ConcatType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_type,
                       std::vector<int> *out_type) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  int dtype = -1;

  for (size_t i = 0; i < in_type->size(); ++i) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    } else {
      CHECK(in_type->at(i) == dtype ||
            in_type->at(i) == -1) <<
          "Non-uniform data type in Concat";
    }
  }

  if (dtype == -1) {
    LOG(FATAL) << "Not enough information to infer type in Concat.";
    return false;
  }

  size_t nin = param_.num_args;
  in_type->clear();
  for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);

  out_type->clear();
  out_type->push_back(dtype);

  return true;
}

inline static bool ConcatForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK(!in_attrs->empty());
  CHECK_EQ(out_attrs->size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask
      // There must be at least one array that are in MKLDNN format.
      && common::ContainsStorage(*in_attrs, kMKLDNNStorage)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[0] = kMKLDNNStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  (*out_attrs)[0] = kDefaultStorage;
  return true;
}

inline static bool BackwardConcatStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
#if MXNET_USE_MKLDNN == 1
  CHECK_EQ(out_attrs->size(), in_attrs->size() - 1);
  if (dev_mask == mshadow::cpu::kDevMask
      && in_attrs->at(0) == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    for (size_t i = 0; i < out_attrs->size(); i++)
      (*out_attrs)[i] = kMKLDNNStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (size_t i = 0; i < out_attrs->size(); i++)
    (*out_attrs)[i] = kDefaultStorage;
  return true;
}

void ConcatComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& op_ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
#if MXNET_USE_MKLDNN == 1
  // MKLDNN support 2D and 4D concat
  if (inputs[0].shape().ndim() == 2 || inputs[0].shape().ndim() == 4) {
    if (inputs[0].dtype() == mshadow::kFloat32) {
      MKLDNNConcat_Forward(attrs, op_ctx, inputs, req, outputs);
    }
  } else {
    std::vector<TBlob> in_blobs(inputs.size());
    for (size_t i = 0; i < in_blobs.size(); i++)
      in_blobs[i] = inputs[i].data();
    std::vector<TBlob> out_blobs(outputs.size());
    for (size_t i = 0; i < out_blobs.size(); i++)
      out_blobs[i] = outputs[i].data();
    ConcatCompute<cpu>(attrs, op_ctx, in_blobs, req, out_blobs);
  }
#endif
}

static void ConcatGradComputeExCPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req, const std::vector<NDArray>& outputs) {
#if MXNET_USE_MKLDNN == 1
  if (inputs[0].shape().ndim() == 2 || inputs[0].shape().ndim() == 4) {
    if (inputs[0].dtype() == mshadow::kFloat32) {
      MKLDNNConcat_Backward(attrs, ctx, inputs, req, outputs);
    }
  } else {
    std::vector<TBlob> in_blobs(1);
    in_blobs[0] = inputs[0].data();
    std::vector<TBlob> out_blobs(outputs.size());
    for (size_t i = 0; i < out_blobs.size(); i++)
      out_blobs[i] = outputs[i].data();
    ConcatGradCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
  }
#endif
}

struct ConcatGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
#if MXNET_USE_MKLDNN == 1
    for (size_t i = 0; i < n->inputs.size(); i++) {
      heads.push_back(n->inputs[i]);
    }
#endif
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(ConcatParam);

NNVM_REGISTER_OP(Concat)
.describe(R"code(Joins input arrays along a given axis.

.. note:: `Concat` is deprecated. Use `concat` instead.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concat(x,y,z,dim=0) = [[ 1.,  1.],
                          [ 2.,  2.],
                          [ 3.,  3.],
                          [ 4.,  4.],
                          [ 5.,  5.],
                          [ 6.,  6.],
                          [ 7.,  7.],
                          [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                         [ 4.,  4.,  7.,  7.],
                         [ 5.,  5.,  8.,  8.]]

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  std::vector<std::string> ret;
  for (int i = 0; i < params.num_args; ++i) {
    ret.push_back(std::string("arg") + std::to_string(i));
  }
  return ret;
})
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr<nnvm::FInferShape>("FInferShape", ConcatShape)
.set_attr<nnvm::FInferType>("FInferType", ConcatType)
.set_attr<FInferStorageType>("FInferStorageType", ConcatForwardInferStorageType)
.set_attr<FCompute>("FCompute<cpu>", ConcatCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConcatComputeExCPU)
.set_attr<nnvm::FGradient>("FGradient", ConcatGrad{"_backward_Concat"})
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(Concat).add_alias("concat");

NNVM_REGISTER_OP(_backward_Concat)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_attr_parser(ParamParser<ConcatParam>)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BackwardConcatStorageType)
.set_attr<FCompute>("FCompute<cpu>", ConcatGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConcatGradComputeExCPU);

}  // namespace op
}  // namespace mxnet

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
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cc
 * \brief CPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile CPU version

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "./sparse_slice-inl.h"

namespace mxnet {
namespace op {

static bool SparseConcatStorage(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(in_attrs->at(0), kDefaultStorage);
  CHECK_EQ(in_attrs->at(1), kDefaultStorage);
  CHECK_EQ(in_attrs->at(2), kRowSparseStorage);
  // dns, ... -> dns
  bool dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                        dispatch_mode, DispatchMode::kFComputeEx);
  return dispatched;
}

static bool SparseConcatShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);
  CHECK_EQ(in_attrs->at(2).ndim(), 2U);
  TShape out_shape(2);
  out_shape[0] = in_attrs->at(1)[0];
  out_shape[1] = in_attrs->at(0)[1];
  out_attrs->at(0) = out_shape;
  return true;
}

static bool SparseConcatType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  out_attrs->at(0) = in_attrs->at(0);
  return true;
}

static bool SparseSliceStorage(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(in_attrs->at(0), kDefaultStorage);
  CHECK_EQ(in_attrs->at(1), kDefaultStorage);
  // dns, ... -> dns
  bool dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                        dispatch_mode, DispatchMode::kFComputeEx);
  return dispatched;
}

static bool SparseSliceShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const SparseSliceParam& params = nnvm::get<SparseSliceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);
  TShape out_shape(2);
  out_shape[0] = params.total_num_rows;
  out_shape[1] = in_attrs->at(0)[1];
  out_attrs->at(0) = out_shape;
  return true;
}

static bool SparseSliceType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  out_attrs->at(0) = in_attrs->at(0);
  return true;
}

struct SparseSlice {
  // copy by rows
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int64_t i, DType* out_dptr, const DType* in_data,
                                  const IType* idx, const nnvm::dim_t num_cols, const nnvm::dim_t upper_num_rows) {
    // adjust idx by the number rows on GPU
    int64_t j = static_cast<int64_t>(idx[i]) - upper_num_rows;
    std::memcpy(out_dptr + i * num_cols, in_data + j * num_cols, num_cols * sizeof(DType));
  }
};


template<>
void SparseSliceComputeRspImpl<cpu>(const SparseSliceParam& params,
                                    const OpContext& ctx,
                                    const TBlob& data,
                                    const TBlob& indices,
                                    const OpReqType req,
                                    const NDArray& out) {
  CHECK_EQ(req, kWriteTo) << "Wrong req";
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  // number of rows stored on GPU
  int64_t upper_num_rows = params.total_num_rows - data.shape_[0];
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(indices.type_flag_, IType, {
      nnvm::dim_t num_cols = data.shape_[1];
      int64_t num_indices = indices.shape_[0];
      const IType *indices_dptr = indices.dptr<IType>();
      // search how many rows are stored on gpu
      auto iter = std::upper_bound(indices_dptr, indices_dptr + num_indices, static_cast<IType>(upper_num_rows));
      nnvm::dim_t offset = iter - indices.dptr<IType>();
      nnvm::dim_t num_output_rows = num_indices - offset;
      const IType *idx_begin = indices_dptr + offset;
      // FIXME num output can be 0
      CHECK_GT(num_output_rows, 0);
      out.CheckAndAlloc({mshadow::Shape1(num_output_rows)});
      // XXX dumpy indices. Not used.
      Fill<false>(s, out.aux_data(kIdx), kWriteTo, 0);
      DType* out_dptr = out.data().dptr<DType>();
      DType* data_dptr = data.dptr<DType>();
      Kernel<SparseSlice, cpu>::Launch(s, num_output_rows, out_dptr, data_dptr, idx_begin, num_cols, upper_num_rows);
    });
  });
}

DMLC_REGISTER_PARAMETER(SparseSliceParam);

NNVM_REGISTER_OP(_contrib_sparse_slice)
.describe(R"code(This operator slices a dense matrix and returns a sparse tensor.

Example::

  // 1 rows stored in gpu
  // 4 rows stored in cpu
  cpu_data=[[1, 8, 4, 8],
            [2, 5, 8, 8],
            [3, 6, 7, 8]]
            [4, 9, 8, 8]]

  v = [0, 1, 4]

  sparse_slice(data, idx, total_num_rows=5).data = 
     [[1, 8, 4, 8],
      [4, 9, 8, 8]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SparseSliceParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FInferStorageType>("FInferStorageType", SparseSliceStorage)
.set_attr<nnvm::FInferShape>("FInferShape", SparseSliceShape)
.set_attr<nnvm::FInferType>("FInferType", SparseSliceType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseSliceComputeEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input embedding")
.add_argument("indices", "NDArray-or-Symbol", "Input indices")
.add_arguments(SparseSliceParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_sparse_concat)
.describe(R"code(This operator slices a dense matrix and returns a sparse tensor.

Example::

  // 1 rows stored in gpu
  gpu_data = [[0, 5, 2, 7]]
  // 4 rows stored in cpu
  cpu_data = [[1, 8, 4, 8],
              [2, 5, 8, 8],
              [3, 6, 7, 8]]
              [4, 9, 8, 8]]

  v = [0, 1, 4]

  slice = sparse_slice(data, idx, total_num_rows=5)
  sparse_concat(gpu_data, idx, slice.copyto(mx.gpu())) = 
     [[0, 5, 2, 7],
     [[1, 8, 4, 8],
      [4, 9, 8, 8]]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices", "cpu_slice"};
  })
.set_attr<FInferStorageType>("FInferStorageType", SparseConcatStorage)
.set_attr<nnvm::FInferShape>("FInferShape", SparseConcatShape)
.set_attr<nnvm::FInferType>("FInferType", SparseConcatType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseConcatComputeEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input embedding")
.add_argument("indices", "NDArray-or-Symbol", "Input indices")
.add_argument("cpu_slice", "NDArray-or-Symbol", "CPU embedding slice");
}
}

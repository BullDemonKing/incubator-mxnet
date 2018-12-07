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

#ifndef MXNET_OPERATOR_CONTRIB_SPARSE_SLICE_INL_H_
#define MXNET_OPERATOR_CONTRIB_SPARSE_SLICE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../channel_op_common.h"
#include "../mxnet_op.h"
#include "../../common/static_array.h"

namespace mxnet {
namespace op {

struct SparseSliceParam : public dmlc::Parameter<SparseSliceParam> {
  int64_t total_num_rows;
  DMLC_DECLARE_PARAMETER(SparseSliceParam) {
    DMLC_DECLARE_FIELD(total_num_rows)
    .describe("Total number of rows");
  }
};  // struct SparseSliceParam

template<typename xpu>
void SparseSliceComputeRspImpl(const SparseSliceParam& params,
                               const OpContext& ctx,
                               const TBlob& data,
                               const TBlob& indices,
                               const OpReqType req,
                               const NDArray& out);

template<typename xpu>
void SparseSliceComputeEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  const SparseSliceParam& params = nnvm::get<SparseSliceParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto data_stype = inputs[0].storage_type();
  const auto idx_stype = inputs[1].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (data_stype == kDefaultStorage && idx_stype == kDefaultStorage &&
      out_stype == kRowSparseStorage) {
    SparseSliceComputeRspImpl<xpu>(params, ctx, inputs[0].data(), inputs[1].data(), req[0], outputs[0]);
  } else {
    LOG(FATAL) << "NOT IMPLEMETNED";
  }
}

struct SparseConcat {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int64_t i, nnvm::dim_t num_rows_to_slice, DType* out_dptr, const DType* in_data,
                                  const DType* in_slice, const IType* idx_ptr, nnvm::dim_t num_cols) {
    nnvm::dim_t row_id = i / num_cols;
    if (row_id < num_rows_to_slice) {
      // read from in_data
      nnvm::dim_t col_id = i % num_cols;
      nnvm::dim_t idx = static_cast<nnvm::dim_t>(idx_ptr[row_id]);
      out_dptr[i] = in_data[idx * num_cols + col_id];
    } else { // read from in_slice
      out_dptr[i] = in_slice[i - num_rows_to_slice * num_cols];
    }
  }
};

template<typename xpu>
void SparseConcatComputeRspImpl(
                                const OpContext& ctx,
                                const TBlob& data,
                                const TBlob& indices,
                                const NDArray& slice,
                                const OpReqType req,
                                const TBlob& out) {
  CHECK_EQ(req, kWriteTo) << "Wrong req";
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(indices.type_flag_, IType, {
      nnvm::dim_t num_cols = data.shape_[1];
      int64_t num_indices = indices.shape_[0];
      CHECK_EQ(num_indices, out.shape_[0]);
      const IType *indices_dptr = indices.dptr<IType>();
      nnvm::dim_t num_sliced_rows = slice.aux_shape(kIdx)[0];
      DType* out_dptr = out.dptr<DType>();
      DType* data_dptr = data.dptr<DType>();
      DType* slice_dptr = slice.data().dptr<DType>();
      // FIXME handle the case where cpu slice is empty
      Kernel<SparseConcat, xpu>::Launch(s, num_indices * num_cols, num_indices - num_sliced_rows, out_dptr,
                                        data_dptr, slice_dptr, indices_dptr, num_cols);
    });
  });
}

template<typename xpu>
void SparseConcatComputeEx(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto data_stype = inputs[0].storage_type();
  const auto idx_stype = inputs[1].storage_type();
  const auto cpu_slice_stype = inputs[2].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (data_stype == kDefaultStorage && idx_stype == kDefaultStorage &&
      cpu_slice_stype == kRowSparseStorage && out_stype == kDefaultStorage) {
    SparseConcatComputeRspImpl<xpu>(ctx, inputs[0].data(), inputs[1].data(),
                                    inputs[2], req[0], outputs[0].data());
  } else {
    LOG(FATAL) << "NOT IMPLEMETNED";
  }
}

}
}

#endif  // MXNET_OPERATOR_CONTRIB_SPARSE_SLICE_INL_H_

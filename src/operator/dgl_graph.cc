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

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "../imperative/imperative_utils.h"
#include "./subgraph_op_common.h"

namespace mxnet {
namespace op {

struct DGLSubgraphParam : public dmlc::Parameter<DGLSubgraphParam> {
  int num_args;
  bool return_mapping;
  DMLC_DECLARE_PARAMETER(DGLSubgraphParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(2)
    .describe("Number of input arguments, including all symbol inputs.");
    DMLC_DECLARE_FIELD(return_mapping)
    .describe("Return mapping of vid and eid between the subgraph and the parent graph.");
  }
};  // struct DGLSubgraphParam

DMLC_REGISTER_PARAMETER(DGLSubgraphParam);

static bool DGLSubgraphStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->at(0), kCSRStorage);
  for (size_t i = 1; i < in_attrs->size(); i++)
    CHECK_EQ(in_attrs->at(i), kDefaultStorage);

  bool success = true;
  *dispatch_mode = DispatchMode::kFComputeEx;
  for (size_t i = 0; i < out_attrs->size(); i++) {
    if (!type_assign(&(*out_attrs)[i], mxnet::kCSRStorage))
    success = false;
  }
  return success;
}

static bool DGLSubgraphShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  for (size_t i = 1; i < in_attrs->size(); i++)
    CHECK_EQ(in_attrs->at(i).ndim(), 1U);

  size_t num_g = params.num_args - 1;
  for (size_t i = 0; i < num_g; i++) {
    TShape gshape(2);
    gshape[0] = in_attrs->at(i + 1)[0];
    gshape[1] = in_attrs->at(i + 1)[0];
    out_attrs->at(i) = gshape;
  }
  for (size_t i = num_g; i < out_attrs->size(); i++) {
    TShape gshape(2);
    gshape[0] = in_attrs->at(i - num_g + 1)[0];
    gshape[1] = in_attrs->at(i - num_g + 1)[0];
    out_attrs->at(i) = gshape;
  }
  return true;
}

static bool DGLSubgraphType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  size_t num_g = params.num_args - 1;
  for (size_t i = 0; i < num_g; i++) {
    CHECK_EQ(in_attrs->at(i + 1), mshadow::kInt64);
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    out_attrs->at(i) = in_attrs->at(0);
  }
  return true;
}

typedef int64_t dgl_id_t;

static void GetSubgraph(const NDArray &csr_arr, const NDArray &varr,
                        const NDArray &sub_csr, const NDArray *old_eids) {
  TBlob data = varr.data();
  int64_t num_vertices = csr_arr.shape()[0];
  const auto len = varr.shape()[0];
  const dgl_id_t *vid_data = data.dptr<dgl_id_t>();
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;
  for (int64_t i = 0; i < len; ++i) {
    oldv2newv[vid_data[i]] = i;
  }

  // Collect the non-zero entries in from the original graph.
  std::vector<dgl_id_t> row_idx(len + 1);
  std::vector<dgl_id_t> col_idx;
  std::vector<dgl_id_t> orig_eids;
  col_idx.reserve(len * 50);
  orig_eids.reserve(len * 50);
  const dgl_id_t *eids = csr_arr.data().dptr<dgl_id_t>();
  const dgl_id_t *indptr = csr_arr.aux_data(csr::kIndPtr).dptr<dgl_id_t>();
  const dgl_id_t *indices = csr_arr.aux_data(csr::kIdx).dptr<dgl_id_t>();
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t oldvid = vid_data[i];
    CHECK_LT(oldvid, num_vertices) << "Vertex Id " << oldvid << " isn't in a graph of "
        << num_vertices << " vertices";
    size_t row_start = indptr[oldvid];
    size_t row_len = indptr[oldvid + 1] - indptr[oldvid];
    // TODO(zhengda) I need to make sure the column index in each row is sorted.
    for (size_t j = 0; j < row_len; ++j) {
      const dgl_id_t oldsucc = indices[row_start + j];
      const dgl_id_t eid = eids[row_start + j];
      auto it = oldv2newv.find(oldsucc);
      if (it != oldv2newv.end()) {
        const dgl_id_t newsucc = it->second;
        col_idx.push_back(newsucc);
        if (old_eids)
          orig_eids.push_back(eid);
      }
    }
    row_idx[i + 1] = col_idx.size();
  }

  TShape nz_shape(1);
  nz_shape[0] = col_idx.size();
  TShape indptr_shape(1);
  indptr_shape[0] = row_idx.size();

  // Store the non-zeros in a subgraph with edge attributes of new edge ids.
  sub_csr.CheckAndAllocData(nz_shape);
  sub_csr.CheckAndAllocAuxData(csr::kIdx, nz_shape);
  sub_csr.CheckAndAllocAuxData(csr::kIndPtr, indptr_shape);
  dgl_id_t *indices_out = sub_csr.aux_data(csr::kIdx).dptr<dgl_id_t>();
  dgl_id_t *indptr_out = sub_csr.aux_data(csr::kIndPtr).dptr<dgl_id_t>();
  std::copy(col_idx.begin(), col_idx.end(), indices_out);
  std::copy(row_idx.begin(), row_idx.end(), indptr_out);
  dgl_id_t *sub_eids = sub_csr.data().dptr<dgl_id_t>();
  for (int64_t i = 0; i < nz_shape[0]; i++)
    sub_eids[i] = i;

  // Store the non-zeros in a subgraph with edge attributes of old edge ids.
  if (old_eids) {
    old_eids->CheckAndAllocData(nz_shape);
    old_eids->CheckAndAllocAuxData(csr::kIdx, nz_shape);
    old_eids->CheckAndAllocAuxData(csr::kIndPtr, indptr_shape);
    dgl_id_t *indices_out = old_eids->aux_data(csr::kIdx).dptr<dgl_id_t>();
    dgl_id_t *indptr_out = old_eids->aux_data(csr::kIndPtr).dptr<dgl_id_t>();
    dgl_id_t *sub_eids = old_eids->data().dptr<dgl_id_t>();
    std::copy(col_idx.begin(), col_idx.end(), indices_out);
    std::copy(row_idx.begin(), row_idx.end(), indptr_out);
    std::copy(orig_eids.begin(), orig_eids.end(), sub_eids);
  }
}

static void DGLSubgraphComputeExCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  size_t num_g = params.num_args - 1;
#pragma omp parallel for
  for (size_t i = 0; i < num_g; i++) {
    const NDArray *old_eids = params.return_mapping ? &outputs[i + num_g] : nullptr ;
    GetSubgraph(inputs[0], inputs[i + 1], outputs[i], old_eids);
  }
}

NNVM_REGISTER_OP(_contrib_dgl_subgraph)
.MXNET_DESCRIBE("")
.set_attr_parser(ParamParser<DGLSubgraphParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  int num_varray = params.num_args - 1;
  if (params.return_mapping)
    return num_varray * 2;
  else
    return num_varray;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const DGLSubgraphParam& params = nnvm::get<DGLSubgraphParam>(attrs.parsed);
  std::vector<std::string> names;
  names.reserve(params.num_args);
  names.push_back("graph");
  for (int i = 1; i < params.num_args; ++i)
    names.push_back("varray" + std::to_string(i - 1));
  return names;
})
.set_attr<FInferStorageType>("FInferStorageType", DGLSubgraphStorageType)
.set_attr<nnvm::FInferShape>("FInferShape", DGLSubgraphShape)
.set_attr<nnvm::FInferType>("FInferType", DGLSubgraphType)
.set_attr<FComputeEx>("FComputeEx<cpu>", DGLSubgraphComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("graph", "Symbol", "Input graph for the message function.")
.add_argument("data", "NDArray-or-Symbol[]",
              "The input arrays that include data arrays and states.")
.add_arguments(DGLSubgraphParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

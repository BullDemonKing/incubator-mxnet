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


///////////////////////// Compact subgraphs ///////////////////////////

struct SubgraphCompactParam : public dmlc::Parameter<SubgraphCompactParam> {
  int num_args;
  bool return_mapping;
  nnvm::Tuple<nnvm::dim_t> graph_sizes;
  DMLC_DECLARE_PARAMETER(SubgraphCompactParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(2)
    .describe("Number of input arguments, including all symbol inputs.");
    DMLC_DECLARE_FIELD(return_mapping)
    .describe("Return mapping of vid and eid between the subgraph and the parent graph.");
    DMLC_DECLARE_FIELD(graph_sizes)
    .describe("the number of vertices in each graph.");
  }
};  // struct SubgraphCompactParam

DMLC_REGISTER_PARAMETER(SubgraphCompactParam);

static inline size_t get_num_graphs(const SubgraphCompactParam &params) {
  // Each CSR needs a 1D array to store the original vertex Id for each row.
  return params.num_args / 2;
}

static void CompactSubgraph(const NDArray &csr, const NDArray &vids,
                            const NDArray &out_csr) {
  TBlob in_idx_data = csr.aux_data(csr::kIdx);
  TBlob in_ptr_data = csr.aux_data(csr::kIndPtr);
  const dgl_id_t *indices_in = in_idx_data.dptr<dgl_id_t>();
  const dgl_id_t *indptr_in = in_ptr_data.dptr<dgl_id_t>();
  const dgl_id_t *row_ids = vids.data().dptr<dgl_id_t>();
  size_t num_elems = csr.aux_data(csr::kIdx).shape_.Size();
  size_t num_vids = vids.shape()[0];
  CHECK_EQ(num_vids, in_ptr_data.shape_[0] - 1);

  // Prepare the Id map from the original graph to the subgraph.
  std::unordered_map<dgl_id_t, dgl_id_t> id_map;
  id_map.reserve(vids.shape()[0]);
  for (size_t i = 0; i < num_vids; i++)
    id_map.insert(std::pair<dgl_id_t, dgl_id_t>(row_ids[i], i));

  TShape nz_shape(1);
  nz_shape[0] = num_elems;
  TShape indptr_shape(1);
  indptr_shape[0] = out_csr.aux_data(csr::kIndPtr).shape_.Size();
  CHECK_GE(in_ptr_data.shape_[0], indptr_shape[0]);

  out_csr.CheckAndAllocData(nz_shape);
  out_csr.CheckAndAllocAuxData(csr::kIdx, nz_shape);
  out_csr.CheckAndAllocAuxData(csr::kIndPtr, indptr_shape);

  dgl_id_t *indices_out = out_csr.aux_data(csr::kIdx).dptr<dgl_id_t>();
  dgl_id_t *indptr_out = out_csr.aux_data(csr::kIndPtr).dptr<dgl_id_t>();
  dgl_id_t *sub_eids = out_csr.data().dptr<dgl_id_t>();
  std::copy(indptr_in, indptr_in + indptr_shape[0], indptr_out);
  for (int64_t i = 0; i < nz_shape[0]; i++) {
    dgl_id_t old_id = indices_in[i];
    auto it = id_map.find(old_id);
    CHECK(it != id_map.end());
    indices_out[i] = it->second;
    sub_eids[i] = i;
  }
}

static void SubgraphCompactComputeExCPU(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  size_t num_g = get_num_graphs(params);
#pragma omp parallel for
  for (size_t i = 0; i < num_g; i++) {
    CompactSubgraph(inputs[0], inputs[i + num_g], outputs[i]);
  }
}

static bool SubgraphCompactStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  size_t num_g = get_num_graphs(params);
  CHECK_EQ(num_g * 2, in_attrs->size());
  // These are the input subgraphs.
  for (size_t i = 0; i < num_g; i++)
    CHECK_EQ(in_attrs->at(i), kCSRStorage);
  // These are the vertex Ids in the original graph.
  for (size_t i = 0; i < num_g; i++)
    CHECK_EQ(in_attrs->at(i + num_g), kDefaultStorage);

  bool success = true;
  *dispatch_mode = DispatchMode::kFComputeEx;
  for (size_t i = 0; i < out_attrs->size(); i++) {
    if (!type_assign(&(*out_attrs)[i], mxnet::kCSRStorage))
      success = false;
  }
  return success;
}

static bool SubgraphCompactShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  size_t num_g = get_num_graphs(params);
  CHECK_EQ(num_g * 2, in_attrs->size());
  // These are the input subgraphs.
  for (size_t i = 0; i < num_g; i++) {
    CHECK_EQ(in_attrs->at(i).ndim(), 2U);
    CHECK_GE(in_attrs->at(i)[0], params.graph_sizes[i]);
    CHECK_GE(in_attrs->at(i)[1], params.graph_sizes[i]);
  }
  // These are the vertex Ids in the original graph.
  for (size_t i = 0; i < num_g; i++) {
    CHECK_EQ(in_attrs->at(i + num_g).ndim(), 1U);
    CHECK_GE(in_attrs->at(i)[0], params.graph_sizes[i]);
  }

  for (size_t i = 0; i < num_g; i++) {
    TShape gshape(2);
    gshape[0] = params.graph_sizes[i];
    gshape[1] = params.graph_sizes[i];
    out_attrs->at(i) = gshape;
    if (params.return_mapping)
      out_attrs->at(i + num_g) = gshape;
  }
  return true;
}

static bool SubgraphCompactType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  for (size_t i = 0; i < in_attrs->size(); i++) {
    CHECK_EQ(in_attrs->at(i), mshadow::kInt64);
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    out_attrs->at(i) = mshadow::kInt64;
  }
  return true;
}

NNVM_REGISTER_OP(_contrib_dgl_graph_compact)
.MXNET_DESCRIBE("")
.set_attr_parser(ParamParser<SubgraphCompactParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  int num_varray = get_num_graphs(params);
  if (params.return_mapping)
    return num_varray * 2;
  else
    return num_varray;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const SubgraphCompactParam& params = nnvm::get<SubgraphCompactParam>(attrs.parsed);
  std::vector<std::string> names;
  names.reserve(params.num_args);
  size_t num_graphs = get_num_graphs(params);
  for (size_t i = 0; i < num_graphs; i++)
    names.push_back("graph" + std::to_string(i));
  for (size_t i = 0; i < num_graphs; ++i)
    names.push_back("varray" + std::to_string(i));
  return names;
})
.set_attr<FInferStorageType>("FInferStorageType", SubgraphCompactStorageType)
.set_attr<nnvm::FInferShape>("FInferShape", SubgraphCompactShape)
.set_attr<nnvm::FInferType>("FInferType", SubgraphCompactType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SubgraphCompactComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("graph_data", "NDArray-or-Symbol[]", "Input graphs and input vertex Ids.")
.add_arguments(SubgraphCompactParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

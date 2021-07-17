#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scheduler_utils {
// Merge all reduction to the right side and returns total number of
// reduction axes
size_t mergeReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (!tv->axis(i)->isReduction() || dont_merge.count(tv->axis(i))) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

// merge all non-reduction axes to the left side and returns total number of
// iteration axes
size_t mergeNonReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge) {
  int prev_i = -1;
  size_t num_merged = 0;
  if (tv->nDims() == 0) {
    return 0;
  }
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (tv->axis(i)->isReduction() || dont_merge.count(tv->axis(i))) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

TensorView* rfactorHelper(
    TensorView* reduction_tv,
    const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(reduction_tv->definition() != nullptr);
  const bool is_welford = reduction_tv->definition()->isA<WelfordOp>();
  if (!is_welford) {
    return reduction_tv->rFactor(axes);
  }
  auto welford = reduction_tv->definition()->as<WelfordOp>();
  auto w_avg = welford->outAvg()->as<TensorView>();
  auto w_var = welford->outVar()->as<TensorView>();
  auto w_n = welford->outN()->as<TensorView>();

  WelfordResult rtvs = reduction_tv->rFactor(axes, w_avg, w_var, w_n);

  // TODO: this can be more generic, using avg because
  //      WelfordOp::out() returns the avg
  return rtvs.avg;
}

namespace {

std::vector<TensorView*> uniqueEntries(
    const std::vector<TensorView*>& tv_deuqe) {
  std::vector<TensorView*> unique_entries;
  std::unordered_set<TensorView*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.emplace(tv_entry).second) {
      unique_entries.emplace_back(tv_entry);
    }
  }
  return unique_entries;
}

} // namespace

std::vector<TensorView*> producerTvsOf(TensorView* tv) {
  if (tv->definition() == nullptr) {
    return {};
  }
  auto producer_vals =
      ir_utils::filterByType<TensorView>(tv->definition()->inputs());
  return uniqueEntries({producer_vals.begin(), producer_vals.end()});
}

std::vector<TensorView*> consumerTvsOf(TensorView* tv) {
  std::vector<TensorView*> consumer_tvs;
  for (auto use_expr : tv->uses()) {
    auto outputs = ir_utils::filterByType<TensorView>(use_expr->outputs());
    consumer_tvs.insert(consumer_tvs.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries(consumer_tvs);
}

std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_producer_tvs;
  for (auto tv : tvs) {
    auto producer_tvs = producerTvsOf(tv);
    all_producer_tvs.insert(
        all_producer_tvs.end(), producer_tvs.begin(), producer_tvs.end());
  }

  return uniqueEntries(all_producer_tvs);
}

std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_consumer_tvs;
  for (auto tv : tvs) {
    auto consumer_tvs = consumerTvsOf(tv);
    all_consumer_tvs.insert(
        all_consumer_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
  }

  return uniqueEntries(all_consumer_tvs);
}

std::vector<TensorView*> inputTvsOf(TensorView* tv) {
  return inputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> outputTvsOf(TensorView* tv) {
  return outputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs) {
  auto inp_vals = IterVisitor::getInputsTo({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(inp_vals);
  std::vector<TensorView*> inp_tvs(filtered.begin(), filtered.end());
  return uniqueEntries(inp_tvs);
}

std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs) {
  auto out_vals = DependencyCheck::getAllOutputsOf({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(out_vals);
  std::vector<TensorView*> out_tvs(filtered.begin(), filtered.end());
  return uniqueEntries(out_tvs);
}

void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs) {
  FusionGuard fg(reference_tv->fusion());

  auto ca_loop_map = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map.build(FusionGuard::getCurFusion());
  for (auto id : reference_tv->domain()->domain()) {
    ca_loop_map.getConcreteMappedID(id)->parallelize(id->getParallelType());
  }

  for (auto tv : all_tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    for (size_t i = 0; i < tv->domain()->domain().size(); i++) {
      tv->axis(i)->parallelize(
          ca_loop_map.getConcreteMappedID(tv->axis(i))->getParallelType());
    }
  }
}

void computeAtInputs(TensorView* consumer, int pos, ComputeAtMode mode) {
  for (auto inp_tv : inputTvsOf(consumer)) {
    inp_tv->computeAt(consumer, pos, mode);
  }
}

void computeWithOutputs(TensorView* producer, int pos, ComputeAtMode mode) {
  for (auto out_tv : outputTvsOf(producer)) {
    producer->computeWith(out_tv, pos, mode);
  }
}

void computeWithOutputs(
    TensorView* producer,
    int pos,
    std::unordered_set<TensorView*> tv_filter,
    ComputeAtMode mode) {
  for (auto out_tv : outputTvsOf(producer)) {
    if (tv_filter.count(out_tv)) {
      producer->computeWith(out_tv, pos, mode);
    }
  }
}

void computeAtOutputs(
    TensorView* producer,
    int pos,
    std::unordered_set<TensorView*> tv_filter,
    ComputeAtMode mode) {
  for (auto out_tv : outputTvsOf(producer)) {
    if (tv_filter.count(out_tv)) {
      producer->computeAt(out_tv, pos, mode);
    }
  }
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  return uniqueEntries({used_tvs.begin(), used_tvs.end()});
}

PersistentBufferInfo persistentBuffers(Fusion* fusion) {
  FusionGuard fg(fusion);

  PersistentBufferInfo info;

  ComputeAtRootDomainMap root_map;
  root_map.build();

  auto all_tvs = allTvs(fusion);

  for (auto producer : all_tvs) {
    bool mappable = true;
    auto consumers = consumerTvsOf(producer);
    if (consumers.empty()) {
      continue;
    }

    auto mappable_roots =
        root_map.getMappableDims(producer->domain(), consumers[0]->domain());

    auto p_root = producer->getMaybeRFactorDomain();

    for (auto p_root_id : p_root) {
      if (p_root_id->isReduction()) {
        continue;
      }
      if (!mappable_roots.count(p_root_id)) {
        mappable = false;
        info.unmappable_dims.emplace(p_root_id);
      }
    }

    if (!mappable) {
      info.buffers.push_back(producer);
    }
  }
  return info;
}

TvProperties getProperties(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* tv) {
  TvProperties properties;
  FusionGuard fg(fusion);

  auto red_root_dom = tv->getRootDomain();
  for (size_t i = red_root_dom.size(); i > 0; i--) {
    if (red_root_dom[i - 1]->isBroadcast()) {
      continue;
    } else if (red_root_dom[i - 1]->isReduction()) {
      break;
    } else {
      properties.fastest_dim_reduction = false;
      break;
    }
  }

  bool hit_reduction = false;
  auto root_dom = tv->getMaybeRFactorDomain();
  for (auto it = root_dom.rbegin(); it != root_dom.rend(); ++it) {
    auto id = *it;

    auto inferred_val = evaluator.evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      hit_reduction = true;
      properties.reduction_numel *= inferred_val.value();
    } else {
      auto dim_size = inferred_val.value();
      properties.iteration_numel *= dim_size;
      if (hit_reduction) {
        properties.iter_outside_red *= dim_size;
      } else {
        properties.iter_inside_red *= dim_size;
      }
    }
  }

  if (properties.reduction_numel == 1) {
    properties.iter_outside_red =
        properties.iter_outside_red * properties.iter_inside_red;
    properties.iter_inside_red = 1;
    properties.fastest_dim_reduction = true;
  }

  return properties;
}

void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& overall_consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<TensorView*> tv_filter,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction) {
  for (auto producer : producers) {
    // Figure out what's between producer and overall_consumers, will not give
    // back any consumers that are not downstream from producer
    auto all_vals_between = DependencyCheck::getAllValsBetween(
        {producer}, {overall_consumers.begin(), overall_consumers.end()});

    std::unordered_set<Val*> all_vals_between_set(
        all_vals_between.begin(), all_vals_between.end());

    for (auto consumer : overall_consumers) {
      if (tv_filter.count(consumer) && all_vals_between_set.count(consumer)) {
        // The way we generate producers and consumers is that we inch away from
        // inputs/outputs. There's a chance we could meet in the middle.
        if (producer == consumer) {
          continue;
        }

        auto pos_it = std::find_if(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end(),
            [&mapped_to_trivial_reduction](IterDomain* id) {
              return mapped_to_trivial_reduction.count(id);
            });

        pos = pos_it == consumer->domain()->domain().end()
            ? pos
            : std::min(
                  (int)std::distance(
                      consumer->domain()->domain().begin(), pos_it) +
                      1,
                  (pos < 0 ? pos + (int)consumer->nDims() : pos));

        // Assume we don't want to reset computeAt on tensors that have already
        // performed it.
        producer->computeAt(consumer, pos, mode);
      }
    }
  }
}

void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& overall_consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction) {
  for (auto producer : producers) {
    // Figure out what's between producer and overall_consumers, will not give
    // back any consumers that are not downstream from producer
    auto all_vals_between = DependencyCheck::getAllValsBetween(
        {producer}, {overall_consumers.begin(), overall_consumers.end()});

    std::unordered_set<Val*> all_vals_between_set(
        all_vals_between.begin(), all_vals_between.end());

    for (auto consumer : overall_consumers) {
      if (all_vals_between_set.count(consumer)) {
        // The way we generate producers and consumers is that we inch away from
        // inputs/outputs. There's a chance we could meet in the middle.
        if (producer == consumer) {
          continue;
        }

        auto pos_it = std::find_if(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end(),
            [&mapped_to_trivial_reduction](IterDomain* id) {
              return mapped_to_trivial_reduction.count(id);
            });

        pos = pos_it == consumer->domain()->domain().end()
            ? pos
            : std::min(
                  (int)std::distance(
                      consumer->domain()->domain().begin(), pos_it) +
                      1,
                  (pos < 0 ? pos + (int)consumer->nDims() : pos));
        // Assume we don't want to reset computeAt on tensors that have already
        // performed it.
        producer->computeAt(consumer, pos, mode);
      }
    }
  }
}

int64_t persistentBufferSize(
    Fusion* fusion,
    torch::jit::fuser::cuda::ExpressionEvaluator& expr_eval) {
  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);

  if (persistent_buffers.buffers.empty()) {
    return true;
  }

  int64_t persistent_buffer_size = 0;

  // Measure at each output how much persistent memory is being used
  std::unordered_map<Val*, int64_t> scoped_persistence;

  for (auto tv : persistent_buffers.buffers) {
    int64_t tv_persistent_numel = -1;
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isReduction()) {
        continue;
      }
      // Unmappable dimensions are those that we cannot inline into other
      // tensor views. So they're the ones that need to be persistent.
      if (!persistent_buffers.unmappable_dims.count(id)) {
        continue;
      }

      auto id_size = expr_eval.evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          id_size.has_value(),
          "Cannot generate heuristics if we don't have input information.");
      if (tv_persistent_numel == -1) {
        tv_persistent_numel = id_size.value();
      } else {
        tv_persistent_numel *= id_size.value();
      }
    }
    persistent_buffer_size =
        tv_persistent_numel * dataTypeSize(tv->getDataType().value());

    // All expressions between tv and its consumers must have tv's persistent
    // buffer allocated. This is an optimistic view on how many registers we
    // need allocated in the kernel, since if we ordered two persistent
    // buffers that are completely independent to somehow overlap with
    // eachother we would assume we wouldn't need those two buffers active at
    // the same time, even though they would be.
    //
    // Unfortunately this limitation is hard to work around as we would have
    // to actually generate the kernel before we know if it would fit
    // persistently in registers. In practice, though, this should not happen
    // as inlining loop structures where the persistent buffer is used should
    // prevent muiltiple persistent buffers from being merged togther if not
    // necessary.
    auto consumers_of_tv = scheduler_utils::consumerTvsOf(tv);
    for (auto val : DependencyCheck::getAllValsBetween(
             {tv}, {consumers_of_tv.begin(), consumers_of_tv.end()})) {
      // Persistent normalization kernels imply that all persistent buffers
      // have the same dimensionality. Assume if a persistent buffer is
      // consumed by another we can alias and reuse the memory.
      if (val == tv) {
        continue;
      }

      if (scoped_persistence.find(val) != scoped_persistence.end()) {
        scoped_persistence.at(val) += persistent_buffer_size;
      } else {
        scoped_persistence[val] = persistent_buffer_size;
      }
    }
  }

  // Find the maximum persistent buffer use
  int64_t max_persistence_size = 0;
  for (auto persistent_entry : scoped_persistence) {
    max_persistence_size =
        std::max(max_persistence_size, persistent_entry.second);
  }

  return max_persistence_size;
}

std::unordered_set<IterDomain*> getTrivialReductionMap(Fusion* fusion) {
  auto all_tvs = allTvs(fusion);
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction;
  for (auto tv : all_tvs) {
    // root domain vs domain shouldn't matter as at this point we shouldn't have
    // any transformations.
    for (auto id : tv->getRootDomain()) {
      if (id->isTrivialReduction()) {
        mapped_to_trivial_reduction.emplace(id);
      }
    }
  }

  if (!mapped_to_trivial_reduction.empty()) {
    // Shouldn't matter which compute at map we use
    auto ca_index_map = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
    ca_index_map.build(fusion);
    // Make a copy we need to check mappings of all
    auto trivial_ids = mapped_to_trivial_reduction;
    for (auto tv : all_tvs) {
      for (auto id : tv->getRootDomain()) {
        if (!id->extent()->isOneInt()) {
          continue;
        }
        if (std::any_of(
                trivial_ids.begin(),
                trivial_ids.end(),
                [&ca_index_map, &id](IterDomain* trivial_id) {
                  return ca_index_map.areMapped(id, trivial_id);
                })) {
          mapped_to_trivial_reduction.emplace(id);
        }
      }
    }
  }
  return mapped_to_trivial_reduction;
}

std::pair<bool, bool> canonicalDimReduction(Fusion* fusion, TensorView* tv) {
  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      scheduler_utils::getTrivialReductionMap(fusion);

  TORCH_INTERNAL_ASSERT(tv != nullptr);

  // We coalesce all reduction axes to the right;
  bool has_red_axis =
      scheduler_utils::mergeReduction(tv, mapped_to_trivial_reduction) > 0;

  bool has_iter_axis =
      scheduler_utils::mergeNonReduction(tv, mapped_to_trivial_reduction) > 0;
  return {has_iter_axis, has_red_axis};
}

TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis) {
  TensorView* reference_tv = nullptr;
  if (rparams.fastest_dim) {
    const int iter_axis = 0;
    const int reduce_axis = has_iter_axis ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      if (rparams.reduction_unroll) {
        // Fastest dim, multiple reductions per block
        // Output Dimensions
        // [x-BIDx, x-TIDy
        //  0       1
        //
        //  Reduction Dimensions
        //  rF-Remain, rf-Unswitch, rf-Unroll, X-TIDx]
        //  2(r)          3(r+1)     4(r+2)    5(r+3)

        //  X-TIDx, rF-Remain, rf-Unswitch, rf-Unroll]
        //   2(r)     3(r+1)       4(r+2)      5(r+3)

        if (!rparams.persistent_kernel) {
          reduction_tv->split(
              reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        } else {
          reduction_tv->split(
              reduce_axis,
              rparams.batches_per_block * rparams.loop_unroll,
              false);
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        }

        reduction_tv->reorder(
            {{reduce_axis + 3, reduce_axis},
             {reduce_axis, reduce_axis + 1},
             {reduce_axis + 1, reduce_axis + 2},
             {reduce_axis + 2, reduce_axis + 3}});

        reference_tv = scheduler_utils::rfactorHelper(
            reduction_tv, {reduce_axis + 1, reduce_axis + 2, reduce_axis + 3});

        reference_tv->axis(reduce_axis)->parallelize(ParallelType::TIDx);
        reference_tv->axis(reduce_axis + 2)
            ->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          reference_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::TIDy);
          if (rparams.split_grid_dim) {
            reference_tv->split(iter_axis, x_grid_limit);
            reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
        reference_tv = reference_tv;
      } else {
        TORCH_INTERNAL_ASSERT(
            has_iter_axis,
            "This scheduler requires an outer dim to the reduction.");
        // Fastest dim, Multiple reductions per block iter unroll
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDy
        //  0       1           2         3
        //
        //  Reduction Dimensions
        //  rF-Remain, r-TIDx]
        //  4(r)     5(r+1)
        if (!rparams.persistent_kernel) {
          reduction_tv->split(
              1, NamedScalar::getParallelDim(ParallelType::TIDx));
        } else {
          reduction_tv->split(1, rparams.batches_per_block, false);
        }

        reference_tv = scheduler_utils::rfactorHelper(reduction_tv, {1});

        reference_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
        reference_tv->split(0, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reference_tv->split(0, 1);

        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDy, rF-Remain, r-TIDx]
        //     0         1          2        3        4         5
        // -> [x-BIDx, x-TIDy, rF-Leftover, x-Unswitch, x-Unroll, r-TIDx]
        //       0        1          2           3          4       5

        reference_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});

        reference_tv->axis(1)->parallelize(ParallelType::TIDy);
        reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
        reference_tv->axis(5)->parallelize(ParallelType::TIDx);

        if (rparams.split_grid_dim) {
          reference_tv->split(0, x_grid_limit);
          reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        }

        reference_tv = reference_tv;
      }
    } else {
      if (rparams.cross_grid) {
        TORCH_INTERNAL_ASSERT(
            rparams.reduction_unroll,
            "Unrolling on iter domain not supported in this scheduler.");

        TORCH_INTERNAL_ASSERT(
            !rparams.persistent_kernel,
            "Grid reductions not implemented yet for persistent kernels.");

        // Fastest dim, cross grid, cross block
        //      [outputs,
        // Idx:     0
        //   | rf-Remain, r-BIDx, r-TIDy, rf-Unswitch, rf-Unroll, r-TIDx]
        //       1(r)     2(r+1)  3(r+2)     4(r+3)      5(r+4)   6(r+5)|
        //                Reduction Dimensions

        //   | r-BIDx, r-TIDy, r-TIDx, rf-Remain, rf-Unswitch, rf-Unroll]
        //       1(r)  2(r+1)  3(r+2)   4(r+3)       5(r+4)     6(r+5)  |
        //                Reduction Dimensions

        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        reduction_tv->split(reduce_axis, rparams.loop_unroll);
        reduction_tv->split(reduce_axis, 1);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::BIDx));

        reduction_tv->reorder(
            {{reduce_axis, reduce_axis + 3},
             {reduce_axis + 1, reduce_axis},
             {reduce_axis + 2, reduce_axis + 1},
             {reduce_axis + 3, reduce_axis + 4},
             {reduce_axis + 4, reduce_axis + 5},
             {reduce_axis + 5, reduce_axis + 2}});

        reference_tv = scheduler_utils::rfactorHelper(
            reduction_tv, {reduce_axis + 3, reduce_axis + 4, reduce_axis + 5});

        reference_tv->axis(reduce_axis + 4)
            ->parallelize(ParallelType::Unswitch);
        reference_tv->axis(reduce_axis + 2)->parallelize(ParallelType::TIDx);
        reference_tv->axis(reduce_axis + 1)->parallelize(ParallelType::TIDy);
        reference_tv->axis(reduce_axis)->parallelize(ParallelType::BIDx);

        if (has_iter_axis) {
          if (rparams.split_grid_dim) {
            reference_tv->split(iter_axis, y_grid_limit);
            reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDy);
          } else {
            reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDy);
          }
        }

        reference_tv = reference_tv;

      } else {
        // Fastest dim, Reduction Splits
        // Output Dimensions
        // [BIDx
        //  0
        //
        // Reduction Dimensions
        // rF-Remain, rf-Unswitch, rf-Unroll, r-TIDx]
        // 1(-4)      2(-3)        3(-2)      4(-1)

        //  r-TIDx, rF-Leftover, rf-Unswitch, rf-Unroll]
        //  1(r)       2(r+1)      3(r+2)       4(r+3)

        if (!rparams.persistent_kernel) {
          reduction_tv->split(
              reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        } else {
          reduction_tv->split(
              reduce_axis,
              rparams.batches_per_block * rparams.loop_unroll,
              false);
          reduction_tv->split(reduce_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(reduce_axis, 1);
        }

        reduction_tv->reorder(
            {{reduce_axis + 3, reduce_axis},
             {reduce_axis, reduce_axis + 1},
             {reduce_axis + 1, reduce_axis + 2},
             {reduce_axis + 2, reduce_axis + 3}});

        reference_tv = scheduler_utils::rfactorHelper(
            reduction_tv, {reduce_axis + 1, reduce_axis + 2, reduce_axis + 3});

        reference_tv->axis(reduce_axis)->parallelize(ParallelType::TIDx);
        reference_tv->axis(reduce_axis + 2)
            ->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          if (rparams.split_grid_dim) {
            reference_tv->split(iter_axis, x_grid_limit);
            reference_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            reference_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }

        reference_tv = reference_tv;
      }
    }
  } else {
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        TORCH_INTERNAL_ASSERT(
            rparams.reduction_unroll,
            "Unrolling on iter domain not supported in this scheduler.");

        TORCH_INTERNAL_ASSERT(
            !rparams.persistent_kernel,
            "Grid reductions not implemented yet for persistent kernels.");

        // Outer Dim, cross grid, cross block

        // Unrolling in this case can only be applied to the reduction dimension
        // since currently, grid reductions cannot be called multiple times
        //
        // Output Dimensions
        // [x-BIDx, x-TIDx,
        //  0         1
        //
        // Reduction Dimensions
        // rF-Leftover, r-BIDy, r-TIDy, rf-Unswitch, rf-Unroll]
        // 2(-5)        3(-4)   4(-3)   5(-2)        6(-1)

        // r-BIDy, r-TIDy, rF-Leftover, rf-Unswitch, rf-Unroll]
        // 2(-5)    3(-4)      4(-3)       5(-2)        6(-1)

        reduction_tv->split(1, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(1, 1);
        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));

        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        reduction_tv->reorder({{2, 4}, {3, 2}, {4, 3}});

        reference_tv = scheduler_utils::rfactorHelper(
            reduction_tv,
            {4, 5, 6}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        reference_tv->axis(5)->parallelize(ParallelType::Unswitch);
        reference_tv->axis(3)->parallelize(ParallelType::TIDy);
        reference_tv->axis(2)->parallelize(ParallelType::BIDy);
        reference_tv->axis(1)->parallelize(ParallelType::TIDx);
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);

        reference_tv = reference_tv;

      } else {
        if (rparams.reduction_unroll || rparams.loop_unroll == 1) {
          // Outer Dim, cross block, unroll reduction dimension

          // Reduction Splits
          // Output Dimensions
          // [x-BIDx, x-TIDx
          //  0       1
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy, rf-Unswitch, rf-Unroll]
          // 2(-4)        3(-3)   4(-2)       5(-1)

          // r-TIDy, rF-Leftover, rf-Unswitch, rf-Unroll]
          // 2(-4)      3(-3)       4(-2)       5(-1)
          if (!rparams.persistent_kernel) {
            reduction_tv->split(1, rparams.loop_unroll);
            // Unswitch axis which gives us finer control on allocations with
            // unrolling
            reduction_tv->split(1, 1);
            reduction_tv->split(
                1, NamedScalar::getParallelDim(ParallelType::TIDy));
          } else {
            reduction_tv->split(1, rparams.batches_per_block, false);
            reduction_tv->split(2, rparams.loop_unroll);
            reduction_tv->split(2, 1);
          }

          reduction_tv->split(
              0, NamedScalar::getParallelDim(ParallelType::TIDx));

          reduction_tv->reorder({{2, 3}, {3, 2}});

          reference_tv = scheduler_utils::rfactorHelper(
              reduction_tv,
              {3, 4, 5}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          reference_tv->axis(4)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(2)->parallelize(ParallelType::TIDy);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);

          reference_tv = reference_tv;

        } else {
          // Outer Dim, cross block, unroll iter dimension

          // Output Dimensions
          // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
          //  0       1           2         3
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy]
          // 4(-2)        5(-1)

          // The unroll/unswitch dimension needs to be within the rF-Leftover
          // dimension
          //    [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, rF-Leftover, r-TIDy]
          //      0(-6)     1(-5)      2(-4)    3(-3)     4(-2)      5(-1)
          // -> [x-BIDx, x-TIDx, rF-Leftover, x-Unswitch, x-Unroll, r-TIDy]
          //      0(-6)   1(-5)     2(-4)        3(-3)      4(-2)    5(-1)

          if (!rparams.persistent_kernel) {
            reduction_tv->split(
                1, NamedScalar::getParallelDim(ParallelType::TIDy));
          } else {
            reduction_tv->split(1, rparams.batches_per_block, false);
          }

          reduction_tv->split(
              0, NamedScalar::getParallelDim(ParallelType::TIDx));
          reduction_tv->split(0, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          reduction_tv->split(0, 1);

          reduction_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});

          reference_tv = scheduler_utils::rfactorHelper(
              reduction_tv,
              {2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          reference_tv->axis(5)->parallelize(ParallelType::TIDy);
          reference_tv->axis(1)->parallelize(ParallelType::TIDx);
          reference_tv->axis(3)->parallelize(ParallelType::Unswitch);
          reference_tv->axis(0)->parallelize(ParallelType::BIDx);

          reference_tv = reference_tv;
        }
      }
    } else {
      if (rparams.reduction_unroll) {
        // Outer Dim, no parallelization on reduction, unroll reduction axis
        // Output Dimensions
        // [x-BIDx, x-TIDx
        //  0       1
        //
        // Reduction Dimensions
        // rf-Leftover, rf-Unswitch, r-Unroll]
        // 2(-3)        3(-2)        4(-1)
        reduction_tv->split(1, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(1, 1);
        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        reduction_tv->axis(0)->parallelize(ParallelType::BIDx);
        reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        reduction_tv->axis(-2)->parallelize(ParallelType::Unswitch);

        reference_tv = reduction_tv;
      } else {
        // No parallelization on reduction, unroll iter axis
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        //
        // Reduction Dimensions
        // r-Leftover]
        // 4(-1)

        // The unroll/unswitch dimension needs to be within the rF-Leftover
        // dimension

        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        reduction_tv->split(0, rparams.loop_unroll);
        reduction_tv->split(0, 1);

        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, r-Leftover]
        //   0(-5)     1(-4)      2(-3)    3(-2)     4(-1)

        reduction_tv->reorder({{1, 3}, {2, 4}, {3, 1}, {4, 2}});

        // [x-BIDx, x-TIDx, r-Leftover, x-Unswitch, x-Unroll]
        //   0(-5)   1(-4)     2(-3)       3(-2)      4(-1)
        reduction_tv->axis(0)->parallelize(ParallelType::BIDx);
        reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        reduction_tv->axis(3)->parallelize(ParallelType::Unswitch);

        reference_tv = reduction_tv;
      }
    }
  }
  return reference_tv;
}

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion) {
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }
}

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
std::vector<TensorView*> cacheInputs(Fusion* fusion, bool unroll) {
  if (!unroll) {
    return {};
  }

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll, make a cache of the inputs
  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  for (auto tv : in_tvs) {
    if (tv->uses().empty()) {
      continue;
    }
    auto cached_tv = tv->cache_after();
    cached_inputs.emplace_back(cached_tv);
  }
  return cached_inputs;
}

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
std::vector<std::pair<TensorView*, TensorView*>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll) {
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  // For intermediate outputs, apply cache_fork
  for (const auto output :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (output->definition() == nullptr) {
      continue;
    }
    if (!output->uses().empty()) {
      if (output->getValType().value() == ValType::TensorView) {
        auto cached_output = output->as<TensorView>()->cache_fork();
        cached_outputs.push_back(std::make_pair(output, cached_output));
      }
    } else if (unroll) {
      auto cached_output = output->as<TensorView>()->cache_before();
      cached_outputs.push_back(std::make_pair(cached_output, output));
    }
  }
  return cached_outputs;
}

void multiReductionInliner(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    std::vector<TensorView*> reduction_tvs,
    std::vector<TensorView*> cached_inputs,
    std::vector<std::pair<TensorView*, TensorView*>> cached_outputs) {
  TransformPropagator::from(reference_tv);

  // Propagate rfactor if necessary
  std::vector<TensorView*> rfactor_tvs;

  if (reference_tv != reduction_tv) {
    std::vector<int> rfactor_axes;
    for (size_t i = 0; i < reference_tv->nDims(); i++) {
      if (reference_tv->axis((int)i)->isReduction() &&
          reference_tv->axis((int)i)->isRFactorProduct()) {
        rfactor_axes.push_back((int)i);
      }
    }

    for (auto reduction_tv_ : reduction_tvs) {
      if (reduction_tv_ == reduction_tv) {
        // The reduction tv
        rfactor_tvs.push_back(reference_tv);
        continue;
      } else {
        rfactor_tvs.push_back(
            scheduler_utils::rfactorHelper(reduction_tv_, rfactor_axes));
      }
    }

    TORCH_INTERNAL_ASSERT(
        reduction_tvs.size() == rfactor_tvs.size(),
        "Expected all reductions to contain rfactor.");
  }

  scheduler_utils::parallelizeAllLike(
      reference_tv, scheduler_utils::allTvs(fusion));

  std::unordered_set<IterDomain*> mapped_to_trivial_reduction =
      scheduler_utils::getTrivialReductionMap(fusion);

  if (rparams.loop_unroll > 1) {
    // Input to cached we want outside unswitched position
    // Cached input to rfactor we want inlined
    std::unordered_set<TensorView*> keep_unrolled;

    std::vector<TensorView*> compute_from;

    // Schedule unrolling on inputs
    for (auto cached_input : cached_inputs) {
      auto consumers_of_input_cache =
          scheduler_utils::consumerTvsOf(cached_input);
      for (auto consumer : consumers_of_input_cache) {
        auto unswitch_it = std::find_if(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end(),
            [&mapped_to_trivial_reduction](IterDomain* id) {
              return id->getParallelType() == ParallelType::Unswitch ||
                  id->getParallelType() == ParallelType::Unroll ||
                  id->getParallelType() == ParallelType::Vectorize ||
                  id->getParallelType() == ParallelType::MisalignedVectorize ||
                  mapped_to_trivial_reduction.count(id);
            });
        auto unswitch_pos = unswitch_it == consumer->domain()->domain().end()
            ? -1
            : std::distance(consumer->domain()->domain().begin(), unswitch_it) +
                1;
        cached_input->computeAt(
            consumer, unswitch_pos, ComputeAtMode::BestEffort);
        compute_from.push_back(consumer);
        keep_unrolled.emplace(cached_input);
      }
    }

    // Make sure not to completely inline if there's trivial reductions in the
    // fusion
    auto pos_it = std::find_if(
        reference_tv->domain()->domain().begin(),
        reference_tv->domain()->domain().end(),
        [&mapped_to_trivial_reduction](IterDomain* id) {
          return mapped_to_trivial_reduction.count(id);
        });

    auto pos = pos_it == reference_tv->domain()->domain().end()
        ? -1
        : std::distance(reference_tv->domain()->domain().begin(), pos_it) + 1;

    // Compute at inputs to rfactor dimensions
    scheduler_utils::computeAtBetween(
        compute_from, rfactor_tvs, pos, ComputeAtMode::MostInlined);

    // Inline rfactor into reduction
    if (reference_tv != reduction_tv) {
      // Compute at rfactor into following reduction, keep outside first
      // reduction iter domain in the rfactor tensor view
      for (size_t i = 0; i < rfactor_tvs.size(); i++) {
        if (!rparams.reduction_unroll) {
          auto rfactor_tv = rfactor_tvs[i];
          auto rfactor_tv_dom = rfactor_tv->domain()->domain();
          auto reduction_it = std::find_if(
              rfactor_tv_dom.begin(), rfactor_tv_dom.end(), [](IterDomain* id) {
                return id->isReduction();
              });
          TORCH_INTERNAL_ASSERT(
              reduction_it != rfactor_tv_dom.end(),
              "Expected reduction axis in ",
              rfactor_tv);
          auto pos = std::distance(rfactor_tv_dom.begin(), reduction_it);
          rfactor_tv->computeWith(
              reduction_tvs[i], pos, ComputeAtMode::Standard);
        } else {
          rfactor_tvs[i]->computeWith(
              reduction_tvs[i], -1, ComputeAtMode::BestEffort);
        }
      }
    }

    // Remove anything before a reduction from compute_from
    {
      auto producers_of_reductions = DependencyCheck::getAllValsBetween(
          {fusion->inputs().begin(), fusion->inputs().end()},
          {reduction_tvs.begin(), reduction_tvs.end()});

      auto producer_tvs_of_reductions =
          ir_utils::filterByType<TensorView>(producers_of_reductions);
      auto compute_from_cleaned = compute_from.erase(
          std::remove_if(
              compute_from.begin(),
              compute_from.end(),
              [&producer_tvs_of_reductions](TensorView* compute_from_tv) {
                return std::find(
                           producer_tvs_of_reductions.begin(),
                           producer_tvs_of_reductions.end(),
                           compute_from_tv) != producer_tvs_of_reductions.end();
              }),
          compute_from.end());
    }

    // Add reduction tensor views to compute from
    compute_from.insert(
        compute_from.end(), reduction_tvs.begin(), reduction_tvs.end());

    std::vector<TensorView*> compute_to;
    for (auto cached_output_pair : cached_outputs) {
      auto cached_output = cached_output_pair.first;
      auto output = cached_output_pair.second;

      // If an output has multiple uses, it should already have computeAt set,
      // don't intefere.
      if (cached_output->uses().size() > 1) {
        continue;
      }

      pos_it = std::find_if(
          output->domain()->domain().begin(),
          output->domain()->domain().end(),
          [&mapped_to_trivial_reduction](IterDomain* id) {
            return id->getParallelType() == ParallelType::Unswitch ||
                id->getParallelType() == ParallelType::Unroll ||
                id->getParallelType() == ParallelType::Vectorize ||
                id->getParallelType() == ParallelType::MisalignedVectorize ||
                mapped_to_trivial_reduction.count(id);
          });
      pos = pos_it == output->domain()->domain().end()
          ? -1
          : std::distance(output->domain()->domain().begin(), pos_it) + 1;

      cached_output->computeAt(output, pos, ComputeAtMode::BestEffort);

      compute_to.push_back(cached_output);
      keep_unrolled.emplace(output);
    }

    scheduler_utils::computeAtBetween(
        compute_from,
        compute_to,
        -1,
        ComputeAtMode::MostInlined,
        mapped_to_trivial_reduction);

  } else {
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly

    for (auto red_tv : reduction_tvs) {
      auto pos_it = std::find_if(
          red_tv->domain()->domain().begin(),
          red_tv->domain()->domain().end(),
          [&mapped_to_trivial_reduction](IterDomain* id) {
            return id->getParallelType() == ParallelType::Unswitch ||
                id->getParallelType() == ParallelType::Unroll ||
                id->getParallelType() == ParallelType::Vectorize ||
                id->getParallelType() == ParallelType::MisalignedVectorize ||
                mapped_to_trivial_reduction.count(id);
          });
      auto pos = pos_it == red_tv->domain()->domain().end()
          ? -1
          : std::distance(red_tv->domain()->domain().begin(), pos_it) + 1;

      scheduler_utils::computeAtInputs(red_tv, pos, ComputeAtMode::MostInlined);
      scheduler_utils::computeWithOutputs(
          red_tv, pos, ComputeAtMode::BestEffort);
    }
  }
}

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Wrapper around set_intersection
template <typename T>
std::set<T> set_intersection(const std::set<T>& set1, const std::set<T>& set2) {
  std::set<T> intersection;
  std::set_intersection(
      set1.begin(),
      set1.end(),
      set2.begin(),
      set2.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection;
}

std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (const auto i : c10::irange(val_chains.size())) {
    auto tv_iterable = ir_utils::filterByType<TensorView>(val_chains[i]);
    tv_chains[i] =
        std::deque<TensorView*>(tv_iterable.begin(), tv_iterable.end());
  }
  return tv_chains;
}

bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= (int)tv->getMaxProducerPosition() &&
      first_mismatch >= (int)tv->getComputeAtPosition();
}

// Return the max position in consumer that producer can be inlined to
// Cannot inline:
//   Reduction dimensions in producer
//   Block broadcast dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
unsigned int getReplayablePosPasC(
    TensorView* producer,
    TensorView* consumer,
    const ComputeAtRootDomainMap& root_map_) {
  // Grab dimensions in producer and consumer that are mappable to eachother
  // based on the computeAtRootDomainMap. This will tell us which dimensions
  // can be inlined based on avoiding trying to inline reduction structures.
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain());

  // Check if any consumer dimensions are marked as vectorize as producer can
  // not be inlined to vectorized dimensions in consumer.
  auto c_dom = consumer->domain()->domain();
  auto vector_dim_it =
      std::find_if(c_dom.begin(), c_dom.end(), [](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType());
      });

  // Limit max position based on vectorized dims in consumer.
  auto max_consumer_pos = std::distance(c_dom.begin(), vector_dim_it);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto c2p_root_map =
      PairwiseRootDomainMap(producer, consumer)
          .mapConsumerToProducer(consumer->domain(), producer->domain());

  auto replay_PasC =
      BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_root_map);

  // Look for id's that map to a consumer id that's vectorized
  auto c2p_replay_map = replay_PasC.getReplay();

  for (size_t consumer_pos = max_consumer_pos; consumer_pos > 0;
       consumer_pos--) {
    auto map_it = c2p_replay_map.find(consumer->axis((int)consumer_pos - 1));
    if (map_it != c2p_replay_map.end()) {
      auto p_id = map_it->second;
      // If we find a consumer dim that maps to a producer dim that's
      // vectorized, or to a producer dim that's a block broadcast, limit max
      // compute at by it
      if (isParallelTypeVectorize(p_id->getParallelType())) {
        max_consumer_pos = consumer_pos - 1;
      }
    }
  }

  // Start at max position and work backwards,  try to find a location where
  // producer can be inlined.
  for (size_t consumer_pos = max_consumer_pos; consumer_pos > 0;
       consumer_pos--) {
    // Grab all root dimensions of consumer as roots must be used to understand
    // inlining potential.
    auto consumer_root_dim_vals =
        IterVisitor::getInputsTo({c_dom.begin(), c_dom.begin() + consumer_pos});
    // convert to iter domains
    auto consumer_root_dim_ids =
        ir_utils::filterByType<IterDomain>(consumer_root_dim_vals);
    // If any root dimensions cannot be mapped to producer we can't inline. If
    // any root dimension
    if (std::any_of(
            consumer_root_dim_ids.begin(),
            consumer_root_dim_ids.end(),
            [&mappable_roots, &c2p_root_map](IterDomain* root_id) {
              return mappable_roots.find(root_id) == mappable_roots.end() &&
                  c2p_root_map.find(root_id) != c2p_root_map.end();
            })) {
      continue;
    }
    return consumer_pos;
  }

  return 0;
}

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Reduction dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
unsigned int getReplayablePosCasP(
    TensorView* consumer,
    TensorView* producer,
    const ComputeAtRootDomainMap& root_map_) {
  // Grab dimensions in producer and consumer that are mappable to eachother
  // based on the computeAtRootDomainMap. This will tell us which dimensions
  // can be inlined based on avoiding trying to inline reduction structures.
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain());

  auto p_dom = producer->domain()->domain();
  auto first_reduction =
      std::find_if(p_dom.begin(), p_dom.end(), [](IterDomain* id) {
        return id->isReduction();
      });

  auto first_vectorized_axis =
      std::find_if(p_dom.begin(), first_reduction, [](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType());
      });

  auto max_producer_pos = std::distance(p_dom.begin(), first_vectorized_axis);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto p2c_root_map = pairwise_root_map.mapProducerToConsumer(
      producer->domain(), consumer->domain());

  auto replay_CasP =
      BestEffortReplay::replayCasP(consumer, producer, -1, pairwise_root_map);

  // Look for id's that map to a consumer id that's vectorized
  auto p2c_replay_map = replay_CasP.getReplay();

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto map_it = p2c_replay_map.find(producer->axis((int)producer_pos - 1));
    if (map_it != p2c_replay_map.end()) {
      auto c_id = map_it->second;
      // If we find a producer dim that maps to a consumer vectorized dim, limit
      // max compute at by it
      if (isParallelTypeVectorize(c_id->getParallelType())) {
        max_producer_pos = producer_pos - 1;
      }
    }
  }

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {producer->getMaybeRFactorDomain().begin(),
         producer->getMaybeRFactorDomain().end()},
        {p_dom.begin(), p_dom.begin() + producer_pos});

    // If any root dims could have mapped to consumer, but don't, then we can't
    // compute at this point
    if (std::any_of(
            producer->getMaybeRFactorDomain().begin(),
            producer->getMaybeRFactorDomain().end(),
            [&mappable_roots, &all_vals](IterDomain* root_id) {
              return std::find(all_vals.begin(), all_vals.end(), root_id) !=
                  all_vals.end() &&
                  mappable_roots.find(root_id) == mappable_roots.end();
            })) {
      continue;
    }

    return producer_pos;
  }
  return 0;
}

} // namespace

void ComputeAt::runAt(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::run");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");

  // Run computeAt on our potentially modified producer(s)
  ComputeAt ca(producer, consumer, consumer, consumer_position, mode);
  ca.runPass();
}

void ComputeAt::runWith(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::runWith");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  ComputeAt ca(producer, consumer, producer, producer_position, mode);
  ca.runPass();
}

// Actually applies transformation
unsigned int ComputeAt::backwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_pos) {
  FUSER_PERF_SCOPE("backwardComputeAt_impl");

  auto max_consumer_compute_at_pos =
      getReplayablePosPasC(producer, consumer, root_map_);
  if (mode_ == ComputeAtMode::BestEffort) {
    consumer_compute_at_pos =
        std::min(consumer_compute_at_pos, max_consumer_compute_at_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    consumer_compute_at_pos = max_consumer_compute_at_pos;
  } else {
    TORCH_INTERNAL_ASSERT(
        consumer_compute_at_pos <= max_consumer_compute_at_pos,
        "Invalid compute at position detected in compute at when trying to replay producer: ",
        producer,
        " as consumer: ",
        consumer,
        " tried to do this at position: ",
        consumer_compute_at_pos,
        " but max position that's allowed is ",
        max_consumer_compute_at_pos);
  }

  auto replay_producer_pair = TransformReplay::replayPasC(
      producer, consumer, (int)consumer_compute_at_pos, root_map_);

  if (replay_producer_pair.second == 0) {
    return 0;
  }

  if (replay_producer_pair.second >= producer->getComputeAtPosition()) {
    const TensorDomain* current_domain = producer->domain();
    TensorDomain* new_domain = replay_producer_pair.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(producer, new_domain),
        "Tried to set the domain of ",
        producer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    producer->setDomain(new_domain);
    if (!producer->isFusionInput()) {
      producer->setComputeAt(replay_producer_pair.second);
      consumer->setMaxProducer(consumer_compute_at_pos);
    }

    root_map_.setAlias(current_domain, new_domain);
  }

  return replay_producer_pair.second;
}

// Actually applies transformation, replay consumer based on producer, set
// compute at of producer, set pass position of consumer, return position
// relative to consumer
unsigned int ComputeAt::forwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_compute_at_pos) {
  FUSER_PERF_SCOPE("forwardComputeAt_impl");

  auto max_producer_compute_at_pos =
      getReplayablePosCasP(consumer, producer, root_map_);

  if (mode_ == ComputeAtMode::BestEffort) {
    producer_compute_at_pos =
        std::min(producer_compute_at_pos, max_producer_compute_at_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    producer_compute_at_pos = max_producer_compute_at_pos;
  } else {
    TORCH_INTERNAL_ASSERT(
        producer_compute_at_pos <= max_producer_compute_at_pos,
        "Invalid compute at position detected in compute at when trying to replay consumer: ",
        consumer,
        " as producer: ",
        producer,
        " tried to do this at position: ",
        producer_compute_at_pos,
        " but max position that's allowed is ",
        max_producer_compute_at_pos);
  }

  auto replay_consumer_pair = TransformReplay::replayCasP(
      consumer, producer, (int)producer_compute_at_pos, root_map_);

  if (producer_compute_at_pos > producer->getComputeAtPosition()) {
    if (!producer->isFusionInput()) {
      producer->setComputeAt((int)producer_compute_at_pos);
    }
  }

  if (replay_consumer_pair.second > consumer->getMaxProducerPosition()) {
    const TensorDomain* current_domain = consumer->domain();
    TensorDomain* new_domain = replay_consumer_pair.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(consumer, new_domain),
        "Tried to set the domain of ",
        consumer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    consumer->setDomain(new_domain);
    if (!producer->isFusionInput()) {
      consumer->setMaxProducer(replay_consumer_pair.second);
    }
    root_map_.setAlias(current_domain, new_domain);
  }

  return replay_consumer_pair.second;
}

void ComputeAt::setCommonConsumer() {
  FUSER_PERF_SCOPE("ComputeAt::setCommonConsumer");

  // Convert the first chain to a set.
  std::set<TensorView*> common_consumers(
      producer_use_chains_.front().begin(), producer_use_chains_.front().end());

  // Run through all use chains of producer, and intersect them to find common
  // TVs
  for (auto tv_chain : producer_use_chains_) {
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(tv_chain.begin(), tv_chain.end()));
  }

  auto all_chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on producer.
  TORCH_CHECK(
      !all_chains.empty(),
      "Compute At expects ",
      producer_->name(),
      " is a dependency of ",
      consumer_->name(),
      ", however it is not.");

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& tv_chain : all_chains) {
    for (auto tv : tv_chain) {
      if (tv != consumer_)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  common_consumer_ = nullptr;
  if (!common_consumers.empty()) {
    for (auto tv : producer_use_chains_.front()) {
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer_ = tv;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        common_consumer_ != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }
}

// Similar to backward traversal in traverseAllKnown but we should only apply
// computeAt if it will increase computeAt positions.
void ComputeAt::traverseBackward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseBackward");
  if (reference_ == producer_) {
    // Forward compute at don't need to run backward traversal
    producer_position_ = reference_position_;
    return;
  }

  // propagate *backward* through all *producer* use_chains or from *producer*
  // to common_consumer if common_consumer exists. Only apply transform if
  // increases computeAt position.
  auto chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  for (auto tv_chain : chains) {
    TensorView* running_producer = tv_chain.back();
    TensorView* running_consumer = nullptr;
    unsigned int running_consumer_pos = reference_position_;
    tv_chain.pop_back();

    TORCH_INTERNAL_ASSERT(running_producer == consumer_);

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      running_consumer_pos = backwardComputeAt_impl(
          running_producer, running_consumer, running_consumer_pos);
    }

    TORCH_INTERNAL_ASSERT(
        running_producer == producer_,
        "Compute at backward traversal ended up on something other than the producer.");
    producer_position_ = running_consumer_pos;
  }
}

void ComputeAt::traverseForward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseForward");

  // propagate forward through all *producer* use_chains or from *producer* to
  // common_consumer if common_consumer exists.
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  // propagate forward through all chains
  for (auto tv_dep_chain : chains) {
    TensorView* running_producer = nullptr;
    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();
    unsigned int running_producer_pos = producer_position_;

    TORCH_INTERNAL_ASSERT(running_consumer == producer_);

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();
      running_producer_pos = forwardComputeAt_impl(
          running_producer, running_consumer, running_producer_pos);
    }
  }
}

namespace {

unsigned int getInnermostNonBroadcastIdFrom(TensorView* tv) {
  unsigned int ret = tv->getComputeAtPosition();

  // Still assuming we only have block broadcast for now.
  //  This part may change
  while (ret > 0 && tv->axis(ret - 1)->isBroadcast()) {
    ret--;
  }

  return ret;
}

// Try to find the aligned position on consumer's domain corresponding to the
//  compute at position of producer domain. Used in computeAt pass only. No
//  checking on actual producer-consumer relationship.
unsigned int getConsumerPosAlignedToProducerCA(
    TensorView* consumer,
    TensorView* producer,
    ComputeAtRootDomainMap& root_map) {
  unsigned int producer_ca_pos = producer->getComputeAtPosition();
  // Locate consumer's position that aligns with
  //  the producer's new compute at axis.
  auto p2c_map = BestEffortReplay::replayCasP(
                     consumer, producer, producer_ca_pos, root_map)
                     .getReplay();

  // Collect the set of iterdomains that are mapped from
  //  producer ids within the compute at pos
  std::unordered_set<IterDomain*> mapped_id_from_producer;
  for (unsigned int producer_i = 0; producer_i < producer_ca_pos;
       producer_i++) {
    auto mapped_it = p2c_map.find(producer->axis(producer_i));
    TORCH_INTERNAL_ASSERT(mapped_it != p2c_map.end());
    mapped_id_from_producer.insert(mapped_it->second);
  }

  // Find the innermost position of consumer that has
  //  been mapped within the producer ca axis.
  unsigned int consumer_pos = consumer->nDims();
  while (consumer_pos > 0 &&
         !mapped_id_from_producer.count(consumer->axis(consumer_pos - 1))) {
    consumer_pos--;
  }

  return consumer_pos;
}

} // namespace

void ComputeAt::hoistInnermostBroadcast() {
  auto fusion = producer_->fusion();

  std::unordered_set<TensorView*> consumers_to_update;

  auto all_vals = fusion->usedMathVals();
  auto all_tvs = ir_utils::filterByType<TensorView>(all_vals);

  for (auto running_producer : all_tvs) {
    if (!running_producer->isFusionInput()) {
      auto producer_ca_pos = running_producer->getComputeAtPosition();
      // Find the innermost iterdomain that is not a broadcast
      auto new_ca_pos = getInnermostNonBroadcastIdFrom(running_producer);
      // Update the compute at pos of this producer if the original
      //  compute at is within inner most broadcast axes
      if (new_ca_pos < producer_ca_pos) {
        running_producer->setComputeAt(new_ca_pos, true);
      }
      // Mark all consumers of this producer for later produce
      //  position update.
      // This is safe with segmented fusion. TV uses will reset
      //  when FusionSegmentGuard try to change the IO.
      for (auto expr_consumer : fusion->unordered_uses(running_producer)) {
        auto tv_consumers =
            ir_utils::filterByType<TensorView>(expr_consumer->outputs());
        consumers_to_update.insert(tv_consumers.begin(), tv_consumers.end());
      }
    }
  }

  // Update the produce positions of all affected consumers
  for (auto running_consumer : consumers_to_update) {
    TORCH_INTERNAL_ASSERT(running_consumer->definition() != nullptr);
    unsigned int new_consummer_pa_pos = 0;

    // Re-compute the max producer position as one or more
    //  of the producers of this consumer have updated their
    //  compute at position.
    for (auto inp : ir_utils::filterByType<TensorView>(
             running_consumer->definition()->inputs())) {
      if (!inp->isFusionInput()) {
        // Locate consumer's position that aligns with
        //  the producer's new compute at axis.
        unsigned int inp_ca_pos_to_consumer =
            getConsumerPosAlignedToProducerCA(running_consumer, inp, root_map_);

        // Populate the max consumer position required by
        //  producer compute at.
        new_consummer_pa_pos =
            std::max(new_consummer_pa_pos, inp_ca_pos_to_consumer);
      }
    }
    // After going through all the producers, decrease the produce
    //  position of current consumer if needed.
    if (new_consummer_pa_pos < running_consumer->getMaxProducerPosition()) {
      running_consumer->setMaxProducer(new_consummer_pa_pos, true);
    }
  }
}

void ComputeAt::runPass() {
  FUSER_PERF_SCOPE("ComputeAt::runPass");

  // Traverse backward through all dep chains from producer to consumer
  traverseBackward();

  // Start at producer and traverse forward through all chains
  traverseForward();

  // Back off on inlining the inner broadcast axes
  hoistInnermostBroadcast();
}

ComputeAt::ComputeAt(
    TensorView* _producer,
    TensorView* _consumer,
    TensorView* _reference,
    unsigned int _reference_position,
    ComputeAtMode _mode)
    : producer_(_producer),
      consumer_(_consumer),
      reference_(_reference),
      reference_position_(_reference_position),
      mode_(_mode) {
  TORCH_INTERNAL_ASSERT(
      reference_ == producer_ || reference_ == consumer_,
      "For compute at reference must be producer or consumer, it's neither.",
      " reference: ",
      reference_,
      " consumer: ",
      consumer_,
      " producer: ",
      producer_);
  TORCH_INTERNAL_ASSERT(
      reference_position_ >= 0 && reference_position_ <= reference_->nDims(),
      "Invalid computeAt axis, received ",
      reference_position_,
      " but should be > -",
      reference_->nDims(),
      " and <= ",
      reference_->nDims(),
      ".");

  producer_use_chains_ = tvChains(DependencyCheck::getAllUseChains(producer_));

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  setCommonConsumer();

  root_map_.build();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

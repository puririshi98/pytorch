#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ExpressionEvaluator;
class SchedulerRuntimeInfo;

namespace scheduler_utils {

constexpr int64_t register_file_size = 256 * 1024;
constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
constexpr int64_t y_grid_limit = 65535;

// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}

// Merge all reduction to the right side and returns total number of
// reduction axes. Don't merge is typically used for trivial reductions.
size_t mergeReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge = {});

// merge all non-reduction axes to the left side and returns total number of
// iteration axes. Don't merge is typically used for trivial reductions.
size_t mergeNonReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge = {});

// Makes rfactor generic with reduction ops and Welford
TensorView* rfactorHelper(TensorView* red_tv, const std::vector<int>& axes);

// Return immediate producers of tv
std::vector<TensorView*> producerTvsOf(TensorView* tv);

// Return immediate consumers of tv
std::vector<TensorView*> consumerTvsOf(TensorView* tv);

// Return immediate producers of tvs (can return tvs input)
std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs);

// Return immediate consumers of tvs (can return tvs input)
std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs);

// Returns producers of tv that are inputs of fusion
std::vector<TensorView*> inputTvsOf(TensorView* tv);

// Returns consumers of tv that are outputs of fusion
std::vector<TensorView*> outputTvsOf(TensorView* tv);

// Returns producers of tvs that are inputs of fusion
std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs);

// Returns consumers of tvs that are outputs of fusion
std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs);

// returns all tensor views in fusion that are used between outputs and inputs.
TORCH_CUDA_CU_API std::vector<TensorView*> allTvs(Fusion* fusion);

TORCH_CUDA_CU_API void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs);

TORCH_CUDA_CU_API void computeAtInputs(
    TensorView* consumer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

TORCH_CUDA_CU_API void computeWithOutputs(
    TensorView* producer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

// compute with outputs if they're present in the provided tv_filter
TORCH_CUDA_CU_API void computeWithOutputs(
    TensorView* producer,
    int pos,
    std::unordered_set<TensorView*> tv_filter,
    ComputeAtMode mode = ComputeAtMode::Standard);

TORCH_CUDA_CU_API void computeAtOutputs(
    TensorView* producer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

// compute at outputs if they're present in the provided tv_filter
TORCH_CUDA_CU_API void computeAtOutputs(
    TensorView* producer,
    int pos,
    std::unordered_set<TensorView*> tv_filter,
    ComputeAtMode mode = ComputeAtMode::Standard);

struct PersistentBufferInfo {
  std::vector<TensorView*> buffers;
  std::unordered_set<IterDomain*> unmappable_dims;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel.
PersistentBufferInfo persistentBuffers(Fusion* fusion);

struct TvProperties {
  // How many elements in tensor view are there to reduce
  int64_t reduction_numel = 1;
  // How many reductions do we need to perform, i.e. how many iter dimension
  // elements are there
  int64_t iteration_numel = 1;
  // Do we reduce the fastest dimension, if no reduction mark true
  bool fastest_dim_reduction = true;
  // What's the iter numel to the left of the reduction (if there is one)
  int64_t iter_outside_red = 1;
  // What's the iter numel to the right of the reduction (if this is or isn't
  // one)
  int64_t iter_inside_red = 1;
};

// Fill TvProperties structure about tv
TvProperties getProperties(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* tv);

// Will call computeAt once on each producer, with the first consumer found that
// is a consumer of the individual producer
void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction = {});

// Will call computeAt once on each producer, with the first consumer found that
// is a consumer of the individual producer
void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<TensorView*> tv_filter,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction = {});

// Compute the amount of register space would be needed to perform this kernel
// persistently, only based on buffers that must be persistent, and based on the
// maximum of all minimum size requirement. i.e. if must be persistent, only
// hold persistent dimension.
int64_t persistentBufferSize(
    Fusion* fusion,
    torch::jit::fuser::cuda::ExpressionEvaluator& expr_eval);

// Returns a set of all iteration domains (in roots of tensors) that map to a
// trivial reduction
std::unordered_set<IterDomain*> getTrivialReductionMap(Fusion* fusion);

// Merges tensor view to the form:
// [IterationDomain, ReductionDomain, TrivialReductionDim0,
// TrivialReductionDim1, ...] Returns if <iteration dimensions, reduction
// dimensions>
std::pair<bool, bool> canonicalDimReduction(Fusion* fusion, TensorView* tv);

// Consistent parallelization based on provided reduction parameters. Provided
// tensor is expected to be reduced by canonicalDimReduction before sending
// here. reduction_tv should be provided as the tensorview to reduce.
// RFactor of reduction_tv will be returned if applicable otherwise reduction_tv
// is returned
TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis);

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion);

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
std::vector<TensorView*> cacheInputs(Fusion* fusion, bool unroll);

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
std::vector<std::pair<TensorView*, TensorView*>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll);

// Inlining function intended for single or multi reduction fusions.
void multiReductionInliner(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    std::vector<TensorView*> reduction_tvs,
    std::vector<TensorView*> cached_inputs,
    std::vector<std::pair<TensorView*, TensorView*>> cached_outputs);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

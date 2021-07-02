#include "utils.h"

#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <sstream>

using namespace torch::jit::fuser::cuda;

std::string toString(ReductionParams rparams) {
  std::stringstream ss;
  if (rparams.fastest_dim) {
    ss << "/Fastest dim";
  } else {
    ss << "/Slow dim";
  }
  if (rparams.cross_block) {
    ss << "/cross block";
  }
  if (rparams.multiple_reds_per_blk) {
    ss << "/multiple reductions per block ";
  }
  if (rparams.cross_grid) {
    ss << "/cross grid";
  }
  if (rparams.loop_unroll > 1) {
    ss << "/Unroll "
       << (rparams.reduction_unroll ? "reduction dim " : "iter dim ")
       << rparams.loop_unroll;
  }
  return ss.str();
}

std::string toString(LaunchParams lparams) {
  std::stringstream ss;
  lparams.toString();
  ss << "/Launch_Parameters["
     << "(" << lparams.bdimz() << "/" << lparams.bdimy() << "/"
     << lparams.bdimx() << ")/(" << lparams.gdimz() << "/" << lparams.gdimy()
     << "/" << lparams.gdimx() << ")/" << lparams.smem() << "]";
  return ss.str();
}

void clearL2Cache() {
  torch::NoGradGuard no_grad;
  auto l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA, 0);

  auto l2_elems = l2_cache_size / 4;
  torch::Tensor t0 = torch::empty(l2_elems, options);
  torch::Tensor t1 = torch::clone(t0);
};

namespace executorCache {
thread_local ExecutorMap executor_map_;
ExecutorMap& getGlobalMap() {
  return executor_map_;
}
} // namespace executorCache

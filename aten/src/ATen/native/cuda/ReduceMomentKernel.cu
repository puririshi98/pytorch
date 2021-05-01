#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

template <typename scalar_t, typename out_t=scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using acc_t = at::acc_type<scalar_t, true>;
  gpu_reduce_kernel<scalar_t, out_t, 2>(iter, WelfordOps<scalar_t, acc_t, int32_t, float, thrust::pair<out_t, out_t>> { unbiased, take_sqrt }, WelfordData<acc_t, int32_t, float> {});
}

static void std_var_kernel_cuda(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  if (iter.dtype(2) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, unbiased, take_sqrt);
  } else if (iter.dtype(2) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, unbiased, take_sqrt);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    iter.dtype(), "std_cuda", [&]() {
      std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, factor_t> {factor});
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if(iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      mean_kernel_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda);

}} // namespace at::native

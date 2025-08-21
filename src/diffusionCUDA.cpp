#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "diffusionCUDAKernel.cuh"

void diffusionCUDA(at::Tensor diams, at::Tensor ratios,
                  at::Tensor ages, at::Tensor hmap, int N, int D) {
    // call to CUDA kernel
    diffusionCUDAKernel(diams.data_ptr<float>(), ratios.data_ptr<float>(), ages.data_ptr<float>(),
                       hmap.data_ptr<float>(), N, D, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("diffusionCUDA", &diffusionCUDA, "Perform crater diffusion using CUDA");
}
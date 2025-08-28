#include "gpu.cuh"
#include <iostream>
#include <cmath>

__global__ void diffuse_k(float *diams, float *ratios, float *ages, float *hmap_in, float *hmap_out, int N, int D) {
    
    // Get indices
    int c = blockIdx.x * blockDim.x + threadIdx.x; // crater index
    if (c >= N) return;

    // Get input values for this crater
    float diam = diams[c];
    float age = ages[c];

    // Compute number of steps
    float k; // m^2 / Myr
    if (diam <= 11.2) {
        k = 0.0155;
    } else if (diam < 45) {
        k = 1.55e-3 * pow(diam, 0.974);
    } else if (diam < 125) {
        k = 1.23e-3 * pow(diam, 0.8386);
    } else {
        k = 5.2e-3 * pow(diam, 1.3);
    }

    // Compute diffusivity, diffusion length scale, and number of steps in time to take
    float kappaT = 1e-6 * k * age;
    float dls = pow((2 * diam / D), 2) / 4;
    int nsteps = ceil(kappaT / dls);

    // Perform diffusion for this crater
    int c_start = c * D * D;
    float dx2 = (diam * 2 / D) * (diam * 2 / D);

    for (int t = 0; t < nsteps; t++){ // time steps
        for (int i = 1; i < D-1; i++){ // rows
            int row_start = c_start + i * D;
            for (int j = 1; j < D-1; j++){ // columns
                int ind = row_start + j;
                float dx = hmap_in[row_start + (j+1)] - 2*hmap_in[ind] + hmap_in[row_start + (j-1)];
                float dy = hmap_in[row_start + D + j] - 2*hmap_in[ind] + hmap_in[row_start - D + j];
                float dd = (dx / dx2) + (dy / dx2);

                // Copy into extra memory so we don't overwrite what we need for future iters
                hmap_out[ind] = hmap_in[ind] + dls * dd;
            }
        }
        // Need to update input map between time step iterations
        for (int i=0; i<D*D; i++){
            hmap_in[c_start+i] = hmap_out[c_start+i];
        }
    }

    // Copy over results and calc min and max depth for d/D
    float max_h = -1000.0;
    float min_h = 1000.0;
    for (int i = 0; i < D * D; i++) {
        if (hmap_out[c_start + i] > max_h) max_h = hmap_out[c_start + i];
        if (hmap_out[c_start + i] < min_h) min_h = hmap_out[c_start + i];
    }

    // Update ratios in place
    ratios[c] = (max_h - min_h) / diam;

}

void diffusionCUDAKernel(float *diams, float *ratios, float *ages, float *hmap, int N, int D, cudaStream_t stream) {

    // Create shared arrays
    float *d_diams, *d_ratios, *d_ages, *d_hmap, *d_domain;
    cudaMalloc(&d_diams, N * sizeof(float));
    cudaMalloc(&d_ratios, N * sizeof(float));
    cudaMalloc(&d_ages, N * sizeof(float));
    cudaMalloc(&d_hmap, N * D * D * sizeof(float));
    cudaMalloc(&d_domain, N * D * D * sizeof(float));

    // Copy data over to shared arrays
    cudaMemcpy(d_diams, diams, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratios, ratios, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ages, ages, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hmap, hmap, N * D * D * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
    diffuse_k<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(d_diams, d_ratios, d_ages, d_hmap, d_domain, N, D);

    // Read the results back into the ratios array
    cudaMemcpy(ratios, d_ratios, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmap, d_domain, N * D * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Error handling
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err){
        std::cout << "CUDA kernel failed with error: " << cudaGetErrorString(err) << std::endl;
    }

    // Clear memory
    cudaFree(d_diams);
    cudaFree(d_ratios);
    cudaFree(d_ages);
    cudaFree(d_hmap);
    cudaFree(d_domain);

    d_diams=NULL;
    d_ratios=NULL;
    d_ages=NULL;
    d_hmap=NULL;
    d_domain=NULL;

}
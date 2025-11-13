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
    int S = ceil(D * (D + 1) / 2); // number of elements in upper triangular vector for one crater profile
    int c_start = c * S;

    // If age is zero, no diffusion necessary so return
    if (fabs(age) < 0.0001) {
        for (int i=0; i < S; i++){
            hmap_out[c_start+i] = hmap_in[c_start+i];
        }
        return;
    }

    float dx2 = (diam * 2 / D) * (diam * 2 / D);
    float dx, dy, dd;
    bool diag;
    int i_map, row, nextx, nexty, prevx, prevy, edge;
    for (int t = 0; t < nsteps; t++){ // time steps
        // for (int i = 1; i < D-1; i++){ // rows
        //     int row_start = c_start + i * D;
        //     for (int j = 1; j < D-1; j++){ // columns
        //         int ind = row_start + j;
        //         float dx = hmap_in[row_start + (j+1)] - 2*hmap_in[ind] + hmap_in[row_start + (j-1)];
        //         float dy = hmap_in[row_start + D + j] - 2*hmap_in[ind] + hmap_in[row_start - D + j];
        //         float dd = (dx / dx2) + (dy / dx2);

        //         // Copy into extra memory so we don't overwrite what we need for future iters
        //         hmap_out[ind] = hmap_in[ind] + dls * dd;
        //     }
        // }
        // only start at (D+1)th cell and end before the last one
        // this skips the first row and the last (this is one element)
        row = 1;
        diag=true;
        edge=(2*D)-2;
        for (int i=D; i<S-1; i++){

            // index for map (accounts for the fact that we are indexing at different locations)
            i_map = c_start + i;

            // if an edge element, skip it
            if (i == edge) {
                diag = true; // next element will be a diagonal element
                row++; // next element will be in next row
                edge += (D-row); // next edge element
                continue;
            }
            // otherwise we can compute the difference
            prevy = i_map-(D-row);
            nextx = i_map+1;
            if (diag) {
                // if diagonal, need to access a different cell for prevx and nexty
                prevx = prevy;
                nexty = nextx;
                diag = false;
            } else {
                prevx = i_map-1;
                nexty = i_map+(D-row-1);
            }
            
            dx = hmap_in[nextx] - 2*hmap_in[i_map] + hmap_in[prevx];
            dy = hmap_in[nexty] - 2*hmap_in[i_map] + hmap_in[prevy];
            dd = (dx / dx2) + (dy / dx2);
            hmap_out[i_map] = hmap_in[i_map] + dls * dd;
        }

        // Update input map between time step iterations
        // Can't do this in normal loop because we need to read values from "future" cells
        // in order to take finite difference of current cell
        for (int i=D; i<S-1; i++){
            hmap_in[c_start+i] = hmap_out[c_start+i];
        }
    }

    // Copy over results and calc min and max depth for d/D
    float max_h = -1000.0;
    float min_h = 1000.0;
    for (int i = 0; i < S; i++) {
        if (hmap_out[c_start + i] > max_h) max_h = hmap_out[c_start + i];
        if (hmap_out[c_start + i] < min_h) min_h = hmap_out[c_start + i];
    }

    // Update ratios in place
    ratios[c] = (max_h - min_h) / diam;

}

void diffusionCUDAKernel(float *diams, float *ratios, float *ages, float *hmap, int N, int D, cudaStream_t stream) {

    // Create shared arrays
    int S = D * (D + 1) / 2;
    float *d_diams, *d_ratios, *d_ages, *d_hmap, *d_domain;
    cudaMalloc(&d_diams, N * sizeof(float));
    cudaMalloc(&d_ratios, N * sizeof(float));
    cudaMalloc(&d_ages, N * sizeof(float));
    cudaMalloc(&d_hmap, N * S * sizeof(float));
    cudaMalloc(&d_domain, N * S * sizeof(float));

    // Copy data over to shared arrays
    cudaMemcpy(d_diams, diams, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratios, ratios, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ages, ages, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hmap, hmap, N * S * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
    diffuse_k<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(d_diams, d_ratios, d_ages, d_hmap, d_domain, N, D);

    // Read the results back into the ratios array
    cudaMemcpy(ratios, d_ratios, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmap, d_domain, N * S * sizeof(float), cudaMemcpyDeviceToHost);

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

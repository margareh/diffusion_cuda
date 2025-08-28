# -*- coding: utf-8 -*-
"""
Performs diffusion of a crater shape using CUDA.

A descriptive guide to the approach used here can be found in
Chapter 7 of Learning Scientific Programming with Python by
Christian Hill, 2nd Edition, 2020.  An online version can be found
at
https://scipython.com/book2/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

# Copyright © 2024, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The “synthterrain” software is licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License
# at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
from diffusionCUDA import diffusionCUDA

def diffusion_cuda(diams, ratios, ages, surfaces, D=200):
    """
    Python wrapper for diffusion model in CUDA
    Inputs: diameters, depth to diameter ratios, and ages for craters; D = domain size
    Outputs: new depth to diameter ratios for each crater (ratios is updated in place by CUDA function)
    """

    # create array to store results in
    # these are all of length N
    diams = torch.tensor(diams).float()
    ratios = torch.tensor(ratios).float()
    ages = torch.tensor(ages).float()
    N = diams.shape[0]

    # sparsify the surfaces (they should be symmetric, only need upper diagonal)
    inds = torch.triu_indices(D, D)
    surfaces_s = torch.tensor(surfaces)[:, inds[0], inds[1]]

    # flatten surfaces
    # this is of length N * S with S = D * (D + 1) / 2
    surfaces_f = surfaces_s.flatten().float()

    # call to CUDA kernel wrapper
    diffusionCUDA(diams, ratios, ages, surfaces_f, N, D)

    # reshape surfaces
    # this also fills in the lower triangle based on the upper triangle
    S = int((D*(D+1)) / 2)
    surfaces = surfaces_f.unflatten(-1, (N, S))
    surfaces_all = torch.zeros((N, D, D))
    surfaces_all[:, inds[0], inds[1]] = surfaces
    surfaces_all[:, inds[1], inds[0]] = surfaces

    surfaces_np = surfaces_all.cpu().numpy()
    ratios_np = ratios.cpu().numpy()

    return ratios_np, surfaces_np

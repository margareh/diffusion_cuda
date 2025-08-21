from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='diffusion_cuda',
    version='0.0',
    packages=find_packages(),
    license='MIT License',
    ext_modules=[
        CUDAExtension(
            name='diffusionCUDA',
            sources=[
                'src/diffusionCUDA.cpp',
                'src/diffusionCUDAKernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
import os
import setuptools

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.2;7.5;8.0"
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile openseg3d!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

setuptools.setup(
    name="openseg3d",
    version="0.0.1",
    author="darrenwang",
    author_email="wangyang9113@gmail.com",
    description="An Open Source Project for 3D Semantic Segmentation",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/WPCLab/OpenSeg3D",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        make_cuda_ext(
            name='voxel_pooling_ext',
            module='seg3d.ops.voxel_pooling',
            extra_include_path=['/usr/local/cuda/include'],
            sources=['src/voxel_pooling.cpp'],
            sources_cuda=['src/voxel_pooling_cuda.cu']),
        make_cuda_ext(
            name='knn_query_ext',
            module='seg3d.ops.knn_query',
            extra_include_path=['/usr/local/cuda/include'],
            sources=['src/knn_query.cpp'],
            sources_cuda=['src/knn_query_cuda.cu']),
        make_cuda_ext(
            name='sampling_ext',
            module='seg3d.ops.sampling',
            extra_include_path=['/usr/local/cuda/include'],
            sources=['src/sampling.cpp'],
            sources_cuda=['src/sampling_cuda.cu']),
        make_cuda_ext(
            name='ingroup_inds_ext',
            module='seg3d.ops.ingroup_inds',
            extra_include_path=['/usr/local/cuda/include'],
            sources=['src/ingroup_inds.cpp'],
            sources_cuda=['src/ingroup_inds_cuda.cu'])
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)

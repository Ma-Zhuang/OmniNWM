from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='occ_pool_ext',
    ext_modules=[
        CUDAExtension(
            name='occ_pool_ext',                 # 编译后的模块名
            sources=['occ_pool.cpp', 'occ_pool_cuda.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
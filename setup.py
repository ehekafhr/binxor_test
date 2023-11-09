from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hdxor',
    ext_modules=[
        CUDAExtension('hdxor', [
            'binary_shared_pytorch.cpp',
            'binary_shared_pytorch_device.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
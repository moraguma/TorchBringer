from setuptools import setup, find_namespace_packages

setup(
    name='torchbringer',
    version='0.1.0',    
    description='A PyTorch library for deep reinforcement learning ',
    url='https://github.com/moraguma/TorchBringer',
    author='Moraguma',
    author_email='g170603@dac.unicamp.br',
    license='MIT',
    packages=find_namespace_packages(),
    install_requires=[
        'setuptools>=70.0.0',
        'torch==2.3.0',
        'gymnasium==0.29.1',
        'aim==3.19.3',
        'numpy',
        'protobuf',
        'grpcio'                    
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',    
        'Environment :: GPU :: NVIDIA CUDA',  
        'Programming Language :: Python'  # TODO : Specify Python versions
    ],
)

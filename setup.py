from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='microcavities',
      author='Yago Del',
      version='1.0',
      description='Microcavities experiments and simulations',
      long_description=long_description,
      url='https://github.com/YagoDel/microcavities',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy<2.0dev',
          'matplotlib<3.0dev',
          'h5py<3.0',
          'pyyaml<6.0',
          'pymsgbox<2.0',
          'nidaqmx<1.0',
          'tmm<1.0'
      ],
      zip_safe=False)

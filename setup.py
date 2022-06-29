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
      install_requires=['numpy', 'matplotlib', 'h5py', 'pyyaml', 'pymsgbox', 'nidaqmx', 'tmm', 'shapely', 'scipy', 'imageio', 'lmfit', 'pyqtgraph'],
      zip_safe=False)

# Taken from
# https://github.com/wichert/pybind11-example/blob/master/setup.py  
# and modified.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from distutils.command.clean import clean as Clean
import os

__version__ = '0.2.1'

install_requires = ['pybind11>=2.5.0', 'numpy>=1.11', 'scipy>=1.0', 'tqdm>=4']

eigen_include_dir = os.environ.get('EIGEN3_INCLUDE_DIR', None)
if eigen_include_dir is None:
    install_requires.append('requests')


class get_eigen_include(object):
    EIGEN3_URL = 'http://bitbucket.org/eigen/eigen/get/3.3.7.zip'
    EIGEN3_DIRNAME = 'eigen-eigen-323c052e1731'
    def __str__(self):

        if eigen_include_dir is not None:
            return eigen_include_dir

        basedir = os.path.dirname(__file__)
        target_dir = os.path.join(basedir, self.EIGEN3_DIRNAME)
        if os.path.exists(target_dir):
            return target_dir

        download_target_dir = os.path.join(basedir, 'eigen3.zip')
        import requests
        import zipfile
        print('Start downloading Eigen library from {}.'.format(self.EIGEN3_DIRNAME))
        response = requests.get(self.EIGEN3_URL, stream=True)
        with open(download_target_dir, 'wb') as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)
        print('Downloaded Eigen into {}.'.format(download_target_dir))

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

headers = [
    'include/myfm/definitions.hpp',
    'include/myfm/util.hpp',
    'include/myfm/FM.hpp',
    'include/myfm/HyperParams.hpp',
    'include/myfm/predictor.hpp',
    'include/myfm/FMTrainer.hpp',
    'include/myfm/FMLearningConfig.hpp',
    'include/myfm/OProbitSampler.hpp',
    'include/Faddeeva/Faddeeva.hh',
    'src/declare_module.hpp'
]


ext_modules = [
    Extension(
        'myfm._myfm',
        ['src/bind.cpp', 'src/Faddeeva.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
            "include"
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-O3'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='myfm',
    version=__version__,
    author='Tomoki Ohtsuki',
    url='https://github.com/tohtsky/myfm',
    author_email='tomoki.ohtsuki129@gmail.com',
    description='Yet another factorization machine',
    long_description='',
    ext_modules=ext_modules,
    install_requires=install_requires,
    setup_requires=install_requires,
    cmdclass={'build_ext': BuildExt},
    packages=['myfm', 'myfm.utils.benchmark_data', 'myfm.utils.callbacks'],
    zip_safe=False,
    headers=headers
)

# Taken from
# https://github.com/wichert/pybind11-example/blob/master/setup.py
# and modified.

import os
import sys
from distutils.ccompiler import CCompiler
from pathlib import Path
from typing import Any, Dict, List

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

install_requires = [
    "numpy>=1.11",
    "scipy>=1.0",
    "tqdm>=4",
    "pandas>=1.0.0",
    "typing-extensions>=4.0.0",
]
setup_requires = ["pybind11>=2.5", "requests", "setuptools_scm"]


eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

TEST_BUILD = os.environ.get("TEST_BUILD", None) is not None


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:

        if eigen_include_dir is not None:
            return eigen_include_dir

        basedir = Path(__file__).resolve().parent
        target_dir = basedir / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return str(target_dir)

        download_target_dir = basedir / "eigen3.zip"
        import zipfile

        import requests

        print("Start downloading Eigen library from {}.".format(self.EIGEN3_DIRNAME))
        response = requests.get(self.EIGEN3_URL, stream=True, verify=False)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)
        print("Downloaded Eigen into {}.".format(download_target_dir))

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return str(target_dir)


class get_pybind_include:
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user: bool = False):
        self.user = user

    def __str__(self) -> str:
        import pybind11

        include_dir: str = pybind11.get_include(self.user)
        return include_dir


headers = [
    "include/myfm/definitions.hpp",
    "include/myfm/util.hpp",
    "include/myfm/FM.hpp",
    "include/myfm/HyperParams.hpp",
    "include/myfm/predictor.hpp",
    "include/myfm/FMTrainer.hpp",
    "include/myfm/FMLearningConfig.hpp",
    "include/myfm/OProbitSampler.hpp",
    "include/Faddeeva/Faddeeva.hh",
    "cpp_source/declare_module.hpp",
]


ext_modules = [
    Extension(
        "myfm._myfm",
        ["cpp_source/bind.cpp", "cpp_source/Faddeeva.cc"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
            "include",
        ],
        language="c++",
    ),
]


def has_flag(compiler: CCompiler, flagname: str) -> bool:
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler: CCompiler) -> str:
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    if TEST_BUILD:
        c_opts: Dict[str, List[str]] = {
            "msvc": ["/EHsc"],
            "unix": ["-O0", "-coverage", "-g"],
        }
        l_opts: Dict[str, List[str]] = {
            "msvc": [],
            "unix": ["-coverage"],
        }
    else:
        c_opts = {
            "msvc": ["/EHsc"],
            "unix": [],
        }
        l_opts = {
            "msvc": [],
            "unix": [],
        }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


def local_scheme(version: Any) -> str:
    return ""


setup(
    name="myfm",
    use_scm_version={"local_scheme": local_scheme},
    author="Tomoki Ohtsuki",
    url="https://github.com/tohtsky/myfm",
    author_email="tomoki.ohtsuki129@gmail.com",
    description="Yet another factorization machine",
    long_description="",
    ext_modules=ext_modules,
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass={"build_ext": BuildExt},
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
    headers=headers,
    python_requires=">=3.6.0",
    package_data={"myfm": ["*.pyi"]},
)

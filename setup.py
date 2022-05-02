import os
from pathlib import Path
from typing import Any

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

install_requires = [
    "numpy>=1.11",
    "scipy>=1.0",
    "tqdm>=4",
    "pandas>=1.0.0",
    "typing-extensions>=4.0.0",
]


TEST_BUILD = os.environ.get("TEST_BUILD", None) is not None


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)
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
    Pybind11Extension(
        "myfm._myfm",
        ["cpp_source/bind.cpp", "cpp_source/Faddeeva.cc"],
        include_dirs=[
            # Path to pybind11 headers
            get_eigen_include(),
            "include",
        ],
    ),
]


def local_scheme(version: Any) -> str:
    return ""


setup(
    name="myfm",
    use_scm_version={"local_scheme": local_scheme},
    author="Tomoki Ohtsuki",
    url="https://github.com/tohtsky/myfm",
    author_email="tomoki.ohtsuki.19937@outlook.jp",
    description="Yet another Bayesian factorization machines.",
    long_description="",
    ext_modules=ext_modules,
    install_requires=install_requires,
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
    zip_safe=False,
    headers=headers,
    python_requires=">=3.6",
    packages=find_packages("src"),
    package_data={"myfm": ["*.pyi"]},
)

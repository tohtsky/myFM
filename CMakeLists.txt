set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

cmake_minimum_required(VERSION 3.0.0)
project(myfm)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -std=c++11 -fPIC")

add_subdirectory(pybind11)
include_directories(include eigen-3.4.0)
pybind11_add_module(_myfm cpp_source/bind.cpp cpp_source/Faddeeva.cc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.0.0)
project(myfm)


add_subdirectory(pybind11)
include_directories(includes eigen-eigen-323c052e1731)
pybind11_add_module(_myfm src/bind.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CPack)

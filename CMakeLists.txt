cmake_minimum_required(VERSION 3.15...3.27)
project(myFM VERSION 0.1.0)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0")
  file(DOWNLOAD https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0.zip")
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/eigen-3.4.0.zip")
endif()

find_package(Threads REQUIRED)


include_directories("${CMAKE_BINARY_DIR}/eigen-3.4.0" cpp_source)
include_directories("${CMAKE_SOURCE_DIR}/include" cpp_source)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# Detect the installed nanobind package and import it into CMake
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_myfm cpp_source/bind.cpp cpp_source/Faddeeva.cc)

install(TARGETS _myfm LIBRARY DESTINATION myfm/)

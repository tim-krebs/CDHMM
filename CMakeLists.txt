cmake_minimum_required(VERSION 3.12)

set(PROJECT "CDHMM")

project("${PROJECT}_Project" VERSION 1.0.0 LANGUAGES CXX)

# Global CMake variables are set here
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set (CONAN_DISABLE_CHECK_COMPILER TRUE)

#Options
option(ENABLE_TESTING   "Enable Unit Testing."  ON)

# Cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
include(Documentation)
if(ENABLE_TESTING)
    include(CTest)
endif()

# Project/Library Names
set(LIBRARY_NAME    "${PROJECT}_MyLib")
set(MAIN            "${PROJECT}_Executable")
set(TEST_MAIN       "${PROJECT}_Unit_test")

#Add conan libraries
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Options for generating/building
# cd build && cmake -D USE_PRINTF=[OFF|ON] ..
option(USE_PRINTF "If you want either to use printf (ON) or cout (OFF)" OFF)

# Configuration Files
configure_file("${PROJECT_SOURCE_DIR}/include/ProjectConfig.h.in"
               "${PROJECT_BINARY_DIR}/ProjectConfig.h" @ONLY)

# Add libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/src/")

# Add executables
add_subdirectory("${PROJECT_SOURCE_DIR}/app/")

# Add tests
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory("${PROJECT_SOURCE_DIR}/tests/")
endif()

project(test)

set(CMAKE_CXX_STANDARD 11)
add_subdirectory(googletest)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../src)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_executable(run_test valueIterationTest.cpp main.cpp testCommon.hpp policyIterationTest.cpp)

target_link_libraries(run_test frozen_lake_lib)

target_link_libraries(run_test gtest)
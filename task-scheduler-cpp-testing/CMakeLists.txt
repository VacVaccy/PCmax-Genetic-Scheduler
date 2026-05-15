cmake_minimum_required(VERSION 3.15)

project(TaskScheduler VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(task_scheduler
    src/main.cpp
    src/Crossing.cpp
    src/Fitness.cpp
    src/GreedyScheduler.cpp
    src/Mutation.cpp
    src/DataParser.cpp
)

target_include_directories(task_scheduler PRIVATE include)
cmake_minimum_required(VERSION 3.18)
project(tensorrt-tool)

add_subdirectory(cxxopts)
add_subdirectory(yolo-tensorrt)

add_executable(tensorrt-tool main.cpp)
target_link_libraries(tensorrt-tool cxxopts detector)
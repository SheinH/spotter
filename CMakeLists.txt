cmake_minimum_required(VERSION 3.20)
project(spotter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # static CRT: nothing extra to load in the game
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_compile_definitions(UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
add_compile_options(/W4 /permissive-)

add_executable(spotter-launch launch/main.cpp)
add_library(spotter SHARED hook/src/spotter.cpp)

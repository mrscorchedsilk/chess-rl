cmake_minimum_required(VERSION 3.18)
project(chess_rl_native LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

enable_testing()

add_library(chess_rl_native_core STATIC
  native/actor.cpp
  native/position.cpp
  native/policy.cpp
  native/node_pool.cpp
  native/mcts.cpp
)
target_compile_features(chess_rl_native_core PUBLIC cxx_std_17)
set_target_properties(chess_rl_native_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(chess_rl_native_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/native"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/third_party/chess-library/include"
)
target_link_libraries(chess_rl_native_core PUBLIC Threads::Threads)

add_executable(native_perft_test native/tests/native_perft_test.cpp)
target_compile_features(native_perft_test PRIVATE cxx_std_17)
set_target_properties(native_perft_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_link_libraries(native_perft_test PRIVATE chess_rl_native_core)
add_test(NAME native_perft_test COMMAND native_perft_test)

add_executable(native_history_test native/tests/native_history_test.cpp)
target_compile_features(native_history_test PRIVATE cxx_std_17)
set_target_properties(native_history_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_link_libraries(native_history_test PRIVATE chess_rl_native_core)
add_test(NAME native_history_test COMMAND native_history_test)

pybind11_add_module(_chess_rl_native MODULE native/chess_rl_native.cpp)
target_compile_features(_chess_rl_native PRIVATE cxx_std_17)
set_target_properties(_chess_rl_native PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  OUTPUT_NAME "_chess_rl_native"
)
target_include_directories(_chess_rl_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/third_party/chess-library/include"
)
target_link_libraries(_chess_rl_native PRIVATE chess_rl_native_core)
target_compile_definitions(_chess_rl_native PRIVATE
  CHESS_RL_NATIVE_VERSION="0.1.0"
  CHESS_RL_NATIVE_ABI_VERSION="1"
  CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT="53e6a841dcda7059a2af363d85f785ef1817304a"
  CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256="f2c8e2e929641e2c71cbe9d8abd718cf3cac46c2a34531215ebd733905e98d7f"
)

install(TARGETS _chess_rl_native
  LIBRARY DESTINATION chess_rl_native
  RUNTIME DESTINATION chess_rl_native
)
install(FILES
  LICENSE
  DESTINATION chess_rl_native
)
install(FILES
  native/third_party/chess-library/LICENSE
  native/third_party/chess-library/PROVENANCE.json
  DESTINATION chess_rl_native/third_party/chess-library
)
install(FILES
  native/third_party/chess-library/include/chess.hpp
  DESTINATION chess_rl_native/third_party/chess-library/include
)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

set(TINYRNN_OPENCL_ACCELERATION TRUE CACHE BOOL "TINYRNN_OPENCL_ACCELERATION")

project(TinyRNN)

if(TINYRNN_OPENCL_ACCELERATION)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
    message("OPENCL_INCLUDE_DIRS: " ${OPENCL_INCLUDE_DIRS})
    message("OPENCL_LIBRARIES: " ${OPENCL_LIBRARIES})

    if(OPENCL_HAS_CPP_BINDINGS)
        message("OpenCL has CPP bindings.")
    else(OPENCL_HAS_CPP_BINDINGS)
        message("No OpenCL CPP bindings found")
    endif(OPENCL_HAS_CPP_BINDINGS)
endif(TINYRNN_OPENCL_ACCELERATION)

get_filename_component(TINY_RNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PATH)
get_filename_component(TINY_RNN_DIR "${TINY_RNN_DIR}" PATH)
set(TINY_RNN_TEST_DIR ${TINY_RNN_DIR}/Tests)

add_definitions(--coverage)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

file(GLOB SOURCES "${TINY_RNN_TEST_DIR}/*.cpp")
list(APPEND SOURCES "${TINY_RNN_TEST_DIR}/ThirdParty/pugixml/src/pugixml.cpp")

# configure the executable

include_directories(${TINY_RNN_DIR}/Source)
add_executable(CatchTests ${SOURCES})

if(TINYRNN_OPENCL_ACCELERATION)
    add_definitions(-DTINYRNN_OPENCL_ACCELERATION=1)
    target_link_libraries(CatchTests ${OPENCL_LIBRARIES})
endif(TINYRNN_OPENCL_ACCELERATION)

# configure unit tests via CTest

enable_testing()
add_test(NAME RunTests COMMAND CatchTests)

add_test(NAME ListTests COMMAND CatchTests --list-tests)
set_tests_properties(ListTests PROPERTIES PASS_REGULAR_EXPRESSION "[0-9]+ test cases")

cmake_minimum_required(VERSION 3.9)

set(ROUTINE H3)

project("${ROUTINE}" LANGUAGES C CXX)
add_executable(${ROUTINE}
    ${PROJECT_SOURCE_DIR}/src/main.cc
)

find_package(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ELSE()
	message(FATAL_ERROR "OpenMP not found.")
ENDIF()


find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
IF(BOOST_FOUND)
	target_link_libraries(${ROUTINE} Boost::program_options)
ELSE()
	message(FATAL_ERROR "BOOST not found.")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O3 -std=c++14 -lrt")

target_include_directories(${ROUTINE} PUBLIC ${PROJECT_SOURCE_DIR}/include)

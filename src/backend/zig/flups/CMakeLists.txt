cmake_minimum_required(VERSION 3.16)
project(ryugu_flups LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(RYUGU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(VENDOR "${RYUGU_ROOT}/C++")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/h3lpr")
file(GLOB H3_HEADERS "${VENDOR}/h3lpr/src/*.hpp")
foreach(HEADER IN LISTS H3_HEADERS)
  get_filename_component(NAME "${HEADER}" NAME)
  configure_file("${HEADER}" "${CMAKE_CURRENT_BINARY_DIR}/include/h3lpr/${NAME}" COPYONLY)
endforeach()
file(GLOB FLUPS_SOURCES "${VENDOR}/flups/src/*.cpp")
list(FILTER FLUPS_SOURCES EXCLUDE REGEX "(_isr|_nb)\\.cpp$")
add_library(flups STATIC ${FLUPS_SOURCES}
  "${VENDOR}/h3lpr/src/macros.cpp"
  "${VENDOR}/h3lpr/src/profiler.cpp")
target_include_directories(flups PRIVATE
  "${RYUGU_ROOT}/src/backend/zig/browser"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${VENDOR}/mpi-serial"
  "${VENDOR}/fftw3-release/api"
  "${VENDOR}/flups/src")
target_compile_definitions(flups PRIVATE NO_BTRACE NDEBUG FORTRAN_UNDERSCORE_ WITHOUT_OPENMP
  FFTW_FLAG=FFTW_ESTIMATE GIT_COMMIT="0c180ba" KERNEL_PATH=".")
set(MPI_SOURCES mpi send recv collective req list handles comm error ic_merge
  group time pack type type_const copy op cart getcount probe info)
foreach(NAME IN LISTS MPI_SOURCES)
  target_sources(flups PRIVATE "${VENDOR}/mpi-serial/${NAME}.c")
endforeach()
target_sources(flups PRIVATE "${RYUGU_ROOT}/src/backend/zig/browser/mpi_fortran.c")

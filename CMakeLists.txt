cmake_minimum_required(VERSION 3.9)

project(TestKokkos CXX)

set (CMAKE_CXX_STANDARD 17)
add_subdirectory(kokkos)
add_subdirectory(YAKL)

set(EXECS mdrange offset_view lr_deep_copy view_rawptr const_t unmanaged_t cmac extent_order subview enable_if_rank mdrp reduce_view multi_reduce hash_view timed_p4 array_to_view minmax random example vector_range)

foreach (EXEC IN LISTS EXECS)
  add_executable(${EXEC} ${EXEC}.cxx)
  target_link_libraries(${EXEC} PUBLIC Kokkos::kokkos yakl)
endforeach()

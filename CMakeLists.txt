cmake_minimum_required(VERSION 3.9)

project(TestKokkos CXX)

set (CMAKE_CXX_STANDARD 17)
add_subdirectory(kokkos)
add_subdirectory(YAKL)
add_executable(mdrange mdrange.cxx)
target_link_libraries(mdrange PUBLIC Kokkos::kokkos yakl)

add_executable(offset_view offset_view.cxx)
target_link_libraries(offset_view PUBLIC Kokkos::kokkos yakl)

add_executable(lr_deep_copy lr_deep_copy.cxx)
target_link_libraries(lr_deep_copy PUBLIC Kokkos::kokkos yakl)

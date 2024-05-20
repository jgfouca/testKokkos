#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "YAKL.h"

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    const int dim0 = 4;
    const int dim1 = 3;
    Kokkos::View<int**> orig_view("testlayout", dim0, dim1);
    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        orig_view(i, j) = i*10 + j;
      }
    }

    Kokkos::View<int**> subv(orig_view, std::make_pair(0, 2), Kokkos::ALL);
    for (size_t i = 0; i < subv.extent(0); ++i) {
      for (size_t j = 0; j < subv.extent(1); ++j) {
        std::cout << "subv(" << i << ", " << j << ") = " << subv(i, j) << std::endl;
      }
    }

    for (auto r = 0; r < 2; ++r) {
      std::cout << "orig DIM " << r << ": " << orig_view.layout().dimension[r] << std::endl;
    }
    for (auto r = 0; r < 2; ++r) {
      std::cout << "subv DIM " << r << ": " << subv.layout().dimension[r] << std::endl;
    }

  }
  Kokkos::finalize();
  return 0;
}

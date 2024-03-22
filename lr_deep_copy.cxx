#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "YAKL.h"

using namespace std;

using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
using HostDevice =
  Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultHostExecutionSpace::memory_space>;

template <typename T, typename Device=DefaultDevice>
using FView = Kokkos::View<T, Kokkos::LayoutLeft, Device>;

template <typename T, typename Device=DefaultDevice>
using CView = Kokkos::View<T, Kokkos::LayoutRight, Device>;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    FView<int**> fview("fview", 2, 3);
    int count = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        fview(i, j) = ++count;
      }
    }

    CView<int**> cview("cview", 2, 3);
    Kokkos::deep_copy(cview, fview);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        std::cout << "fview = " << fview(i, j) << std::endl;
      }
    }
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        std::cout << "cview = " << cview(i, j) << std::endl;
      }
    }

    for (size_t i = 0; i < fview.size(); ++i) {
      std::cout << fview.data()[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < cview.size(); ++i) {
      std::cout << cview.data()[i] << " ";
    }
    std::cout << std::endl;

  }
  Kokkos::finalize();
  return 0;
}

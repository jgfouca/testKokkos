#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

template <typename ViewT, typename std::enable_if<ViewT::rank == 1>::type* = nullptr>
void func(const ViewT& view)
{
  for (size_t i = 0; i < view.extent(0); ++i) {
    std::cout << view(i) << " ";
  }
  std::cout << std::endl;
}

template <typename ViewT, typename std::enable_if<ViewT::rank == 2>::type* = nullptr>
void func(const ViewT& view)
{
  for (size_t i = 0; i < view.extent(0); ++i) {
    for (size_t j = 0; j < view.extent(0); ++j) {
      std::cout << view(i, j) << " ";
    }
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    const int dim = 6;
    const int dim0 = 3;
    const int dim1 = 4;
    Kokkos::View<int**> orig_view("testlayout", dim0, dim1);
    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        orig_view(i, j) = i*10 + j;
      }
    }

    func(orig_view);

    Kokkos::View<int*> odv("asdsad", dim);
    Kokkos::deep_copy(odv, dim);

    func(odv);

  }
  Kokkos::finalize();
  return 0;
}

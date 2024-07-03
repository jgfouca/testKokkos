#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

using view_t = Kokkos::View<int**>;
using uview_t = Unmanaged<view_t>;
using aview_t = Atomic<uview_t>;

void func1(const view_t& arg)
{
  for (size_t i = 0; i < arg.size(); ++i) {
    std::cout << arg.data()[i] << " ";
  }
  std::cout << std::endl;
}

void func2(const uview_t& arg)
{
  for (size_t i = 0; i < arg.size(); ++i) {
    std::cout << arg.data()[i] << " ";
  }
  std::cout << std::endl;
}


int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    view_t data("data", 2, 4);
    uview_t data2(data.data(), 2, 4);
    aview_t data3(data.data(), 2, 4);
    Kokkos::deep_copy(data, 42);
    func1(data);
    func2(data);
    func1(data2);
    func2(data2);
    func1(data3);
    func2(data3);
  }
  Kokkos::finalize();
  return 0;
}

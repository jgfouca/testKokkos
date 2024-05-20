#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    Kokkos::View<int**> view1("view1", 10, 20);
    std::cout << "extent0: " << view1.extent(0) << std::endl;
    std::cout << "extent1: " << view1.extent(1) << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

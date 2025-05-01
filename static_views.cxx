#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    static Kokkos::View<int*> view1("view1", 10);
    Kokkos::deep_copy(view1, 1);

    view1 = decltype(view1)();
  }
  Kokkos::finalize();
  return 0;
}

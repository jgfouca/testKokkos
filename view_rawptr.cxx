#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    int* raw = new int[10];

    {
      Kokkos::View<int*> view1(raw, 10);
      for (int i = 0; i < 10; ++i) view1(i) = i+1;
    }

    {
      Kokkos::View<int*> view2(raw, 10);
      for (int i = 0; i < 10; ++i) view2(i) = i+2;
    }

    for (int i = 0; i < 10; ++i) raw[i] = i+3;

    delete[] raw;
  }
  Kokkos::finalize();
  return 0;
}

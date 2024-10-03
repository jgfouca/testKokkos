#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    constexpr int dim1 = 2;
    constexpr int dim2 = 3;

    int carray[dim1][dim2] = {
      {1, 2, 3},
      {4, 5, 6}
    };

    Kokkos::View<int**> view(&carray[0][0], dim1, dim2);
    std::cout << "Validating" << std::endl;
    for (int k = 0; k < dim1; ++k) {
      for (int j = 0; j < dim2; ++j) {
        std::cout << carray[k][j] << " == " << view(k, j) << std::endl;
      }
    }
  }
  Kokkos::finalize();
  return 0;
}

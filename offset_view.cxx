#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "YAKL.h"

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    auto rg1 = std::make_pair(-1, 2);
    auto rg2 = std::make_pair(-2, 1);
    Kokkos::Experimental::OffsetView<int**> oview2("testlayout", rg1, rg2);
    Kokkos::deep_copy(oview2, 42);
    int count = 0;
    for (int i = rg1.first; i <= rg1.second; ++i) {
      for (int j = rg2.first; j <= rg2.second; ++j, ++count) {
        std::cout << "oview2(" << i << ", " << j << ") = " << oview2(i, j) << std::endl;
      }
    }
    std::cout << oview2.size() << " " << count << std::endl;

    Kokkos::Experimental::OffsetView<int*> oview1("testlayout2", std::make_pair(0, 10));
    std::cout << oview1.size() << std::endl;

    Kokkos::View<int***> view3d("test", 1, 2, 3);
    std::cout << "Is offset view a view? " << Kokkos::is_view<decltype(oview1)>::value << std::endl;
    std::cout << "Is view a view? " << Kokkos::is_view<decltype(view3d)>::value << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

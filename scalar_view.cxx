#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

using view_t = Kokkos::View<int>;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    view_t view1("view1");

    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int) {
        view1() = 42;
    });

    auto view1m = Kokkos::create_mirror_view(view1);
    Kokkos::deep_copy(view1m, view1);
    cout << view1m() << endl;
  }
  Kokkos::finalize();
  return 0;
}

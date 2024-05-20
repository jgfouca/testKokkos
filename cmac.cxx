#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    Kokkos::View<int*> view1("view1", 10);
    Kokkos::deep_copy(view1, 1);

    auto view2 = Kokkos::create_mirror_view_and_copy(DefaultDevice(), view1);
    for (size_t i = 0; i < view2.size(); ++i) {
      std::cout << view2(i) << " ";
    }
    std::cout << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

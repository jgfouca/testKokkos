#include <iostream>
#include <string>
#include <chrono>
#include <Kokkos_Core.hpp>

using namespace std;

#define TIMED_P4(kernel)                  \
  {                                             \
  auto start_t = std::chrono::high_resolution_clock::now();     \
  kernel;                                                               \
  auto stop_t = std::chrono::high_resolution_clock::now();              \
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t); \
  static double total_s = 0.; \
  total_s += duration.count() / 1000000.0; \
  std::cout << "For file " << __FILE__ << ", line " << __LINE__ << ", total is: " << total_s << " s" << std::endl;   \
  }

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {

    Kokkos::View<int*> view("view", 4);
    Kokkos::deep_copy(view, 4);

    for (int j = 0; j < 4; ++j) {
      TIMED_P4(Kokkos::parallel_for(4, KOKKOS_LAMBDA(int i) {
            cout << view(i) << endl;
          }));
    }
  }
  Kokkos::finalize();
  return 0;
}

#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

static constexpr int DIM = 3;

using view_t = Kokkos::View<int*>;
using pair_t = Kokkos::pair<int, int>;

pair_t& operator+=(pair_t& lhs, const pair_t& rhs)
{
  lhs.first += rhs.first;
  lhs.second += rhs.second;
  return lhs;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    view_t view1("view1", DIM);

    for (int i = 0; i < DIM; ++i) {
      view1(i) = i+1;
    }

    pair_t sum = {0, 0};
    // Doesn't work using pairs directly
    // Kokkos::parallel_reduce(DIM, KOKKOS_LAMBDA(const int i, pair_t& sum_inner) {
    //   const int item = view1(i);
    //   sum_inner.first += item;
    //   if (item == 2) {
    //     sum_inner.second = i;
    //   }
    // }, Kokkos::Sum<pair_t>(sum));

    Kokkos::parallel_reduce(DIM, KOKKOS_LAMBDA(const int i, int& sum_inner1, int& sum_inner2) {
      const int item = view1(i);
      sum_inner1 += item;
      if (item == 2) {
        sum_inner2 = i;
      }
      }, sum.first, sum.second);

    std::cout << "First run: " << std::endl;
    std::cout << "Sum is: " << sum.first  << std::endl;
    std::cout << "Idx is: " << sum.second << std::endl;

    Kokkos::parallel_reduce(DIM, KOKKOS_LAMBDA(const int i, int& sum_inner1, int& sum_inner2) {
      const int item = view1(i);
      sum_inner1 += item;
      if (item == 2) {
        sum_inner2 = i;
      }
    }, sum.first, sum.second);

    std::cout << "First run: " << std::endl;
    std::cout << "Sum is: " << sum.first  << std::endl;
    std::cout << "Idx is: " << sum.second << std::endl;

  }
  Kokkos::finalize();
  return 0;
}

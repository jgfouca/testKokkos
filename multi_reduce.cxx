#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

static constexpr int DIM = 3;
static constexpr int DIM2 = 2;

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

    // Now let's try a multireduce in hierarchical
    using team_policy_t = Kokkos::TeamPolicy<>;
    using member_type_t = team_policy_t::member_type;
    team_policy_t team_policy(DIM, 1);

    Kokkos::View<int**> view2("view1", DIM, DIM2);
    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM2; ++j) {
        view2(i, j) = j+10 + i;
      }
    }

    Kokkos::View<int*> view3("sums", DIM);
    Kokkos::parallel_for("HierarchicalLoop", team_policy, KOKKOS_LAMBDA(const member_type_t& member) {
      const int i = member.league_rank();
      int sum1, sum2, sum3;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, DIM2), [=](const int k, int& lsum1, int& lsum2, int& lsum3) {
        lsum1 += view2(i, k);
        lsum2 += view2(i, k) + 1;
        lsum3 += view2(i, k) + 2;
      }, sum1, sum2, sum3);
      view3(i) = sum1 + sum2 + sum3;
    });

    for (int i = 0; i < DIM; ++i) {
      std::cout << "Sum of dim " << i << " = " << view3(i) << std::endl;
    }

  }
  Kokkos::finalize();
  return 0;
}

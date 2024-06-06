#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

template <typename LayoutT, typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
struct MDRP
{
  static constexpr Kokkos::Iterate LeftI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Left
    : Kokkos::Iterate::Right;
  static constexpr Kokkos::Iterate RightI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Right
    : Kokkos::Iterate::Left;

  template <int Rank>
  using MDRP_t = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<Rank, LeftI, RightI> >;

  template <int N, typename IntT>
  static inline
  MDRP_t<N> get(const IntT (&upper_bounds)[N])
  {
    assert(N > 1);
    const IntT lower_bounds[N] = {0};
    return MDRP_t<N>(lower_bounds, upper_bounds); //, DefaultTile<N>::value);
  }
};

template <typename ViewT>
void func(const ViewT& view)
{
  using LayoutT = typename ViewT::array_layout;
  using MDRP_t  = MDRP<LayoutT>;

  Kokkos::parallel_for( MDRP_t::template get<2>({view.extent(0),view.extent(1)}), KOKKOS_LAMBDA (int i, int j) {
      std::cout << view(i, j) << " ";
  });
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    const int dim0 = 3;
    const int dim1 = 4;
    Kokkos::View<int**> orig_view("testlayout", dim0, dim1);
    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        orig_view(i, j) = i*10 + j;
      }
    }

    func(orig_view);
  }
  Kokkos::finalize();
  return 0;
}

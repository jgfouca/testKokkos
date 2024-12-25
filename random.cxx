#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

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

class Random {
 protected:
  /** @private */
  typedef unsigned long long u8;
  /** @private */
  u8 static constexpr rot(u8 x, u8 k) { return (((x)<<(k))|((x)>>(64-(k)))); }
  /** @private */
  struct State { u8 a, b, c, d; };
  /** @private */
  State state;

 public:

  /** @brief Initializes a prng object with the seed 1368976481. Warm-up of 20 iterations. */
  KOKKOS_INLINE_FUNCTION Random()                            { set_seed(1368976481L); } // I made up this number
  /** @brief Initializes a prng object with the specified seed. Warm-up of 20 iterations. */
  KOKKOS_INLINE_FUNCTION Random(u8 seed)                     { set_seed(seed); }
  /** @brief Copies a Random object */
  KOKKOS_INLINE_FUNCTION Random(Random const            &in) { this->state = in.state; }
  /** @brief Moves a Random object */
  KOKKOS_INLINE_FUNCTION Random(Random                 &&in) { this->state = in.state; }
  /** @brief Copies a Random object */
  KOKKOS_INLINE_FUNCTION Random &operator=(Random const &in) { this->state = in.state; return *this; }
  /** @brief Moves a Random object */
  KOKKOS_INLINE_FUNCTION Random &operator=(Random      &&in) { this->state = in.state; return *this; }

  /** @brief Assigns a seed. Warm-up of 20 iterations. */
  KOKKOS_INLINE_FUNCTION void set_seed(u8 seed) {
    state.a = 0xf1ea5eed;  state.b = seed;  state.c = seed;  state.d = seed;
    for (int i=0; i<20; ++i) { gen(); }
  }

  /** @brief Generates a random unsigned integer between zero and `std::numeric_limits<u8>::max() - 1` */
  KOKKOS_INLINE_FUNCTION u8 gen() {
    u8 e    = state.a - rot(state.b, 7);
    state.a = state.b ^ rot(state.c,13);
    state.b = state.c + rot(state.d,37);
    state.c = state.d + e;
    state.d = e       + state.a;
    return state.d;
  }

  /** @brief Generates a random floating point value between `0` and `1`
   * @param T The type of the floating point number */
  template <class T> KOKKOS_INLINE_FUNCTION T genFP() {
    return static_cast<T>(gen()) / static_cast<T>(std::numeric_limits<u8>::max());
  }

  /** @brief Generates a random floating point value between `lb` and `ub`
   * @param T  The type of the floating point number
   * @param lb Lower bound of the random number
   * @param ub Upper bound of the random number*/
  template <class T> KOKKOS_INLINE_FUNCTION T genFP(T lb, T ub) {
    return genFP<T>() * (ub-lb) + lb;
  }

};

template <typename ViewT>
void func(const ViewT& view)
{
  using LayoutT = typename ViewT::array_layout;
  using MDRP_t  = MDRP<LayoutT>;

  const uint64_t seed = 12345;

  Kokkos::Random_XorShift64_Pool<> rand_pool(seed);

  Kokkos::parallel_for( MDRP_t::template get<2>({view.extent(0),view.extent(1)}), KOKKOS_LAMBDA (int i, int j) {
    auto generator = rand_pool.get_state();
    view(i, j) = generator.drand(0., 1.);
    rand_pool.free_state(generator);
  });
}

template <typename ViewT>
void func2(const ViewT& view)
{
  using LayoutT = typename ViewT::array_layout;
  using MDRP_t  = MDRP<LayoutT>;

  const uint64_t seed = 12345;

  Kokkos::parallel_for( MDRP_t::template get<2>({view.extent(0),view.extent(1)}), KOKKOS_LAMBDA (int i, int j) {
    Random rand(seed + i*10 + j);
    view(i, j) = rand.genFP<typename ViewT::non_const_value_type>();
  });
}



int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    const int dim0 = 8;
    const int dim1 = 4;
    Kokkos::View<double**> orig_view("testlayout", dim0, dim1);

    //func(orig_view);
    func2(orig_view);
    Kokkos::fence();

    auto hostm = Kokkos::create_mirror_view(orig_view);
    Kokkos::deep_copy(hostm, orig_view);

    for (size_t i = 0; i < hostm.extent(0); ++i) {
      for (size_t j = 0; j < hostm.extent(1); ++j) {
        std::cout << hostm(i, j) << std::endl;
      }
    }
  }
  Kokkos::finalize();
  return 0;
}

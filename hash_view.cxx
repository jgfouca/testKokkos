#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

static constexpr int DIM = 3;

using view_t = Kokkos::View<int*>;

using HashType = std::uint64_t;

template <typename View>
struct HashReducer {
  using reducer = HashReducer ;
  using value_type = HashType;
  using exe_space = typename View::execution_space;
  using result_view_type = Kokkos::View<value_type*, exe_space, Kokkos::MemoryUnmanaged>;

  KOKKOS_INLINE_FUNCTION
  static void hash (const HashType v, HashType& accum) {
    constexpr auto first_bit = 1ULL << 63;
    accum += ~first_bit & v; // no overflow
    accum ^=  first_bit & v; // handle most significant bit
  }

  KOKKOS_INLINE_FUNCTION
  static void hash (const double v_, HashType& accum) {
    static_assert(sizeof(double) == sizeof(HashType),
                  "HashType must have size sizeof(double).");
    HashType v;
    std::memcpy(&v, &v_, sizeof(HashType));
    hash(v, accum);
  }

  KOKKOS_INLINE_FUNCTION
  static void hash (const float v, HashType& accum) {
    hash(double(v), accum);
  }

  KOKKOS_INLINE_FUNCTION
  static void hash (const int v, HashType& accum) {
    hash(double(v), accum);
  }

  KOKKOS_INLINE_FUNCTION HashReducer (value_type& value_) : value(value_) {}
  KOKKOS_INLINE_FUNCTION void join (value_type& dest, const value_type& src) const { hash(src, dest); }
  KOKKOS_INLINE_FUNCTION void init (value_type& val) const { val = 0; }
  KOKKOS_INLINE_FUNCTION value_type& reference () const { return value; }
  KOKKOS_INLINE_FUNCTION bool references_scalar () const { return true; }
  KOKKOS_INLINE_FUNCTION result_view_type view () const { return result_view_type(&value, 1); }

private:
  value_type& value;
};

template <typename View>
HashType hash_view (const View& v) {
  using reducer = HashReducer<View>;
  HashType accum_out = 0;
  HashType accum = 0;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<typename View::execution_space>(0, v.size()),
    KOKKOS_LAMBDA(const int idx, HashType& accum) {
      reducer::hash(v(idx), accum);
    }, reducer(accum));
  Kokkos::fence();
  reducer::hash(accum, accum_out);
  std::cout << "Hash is: " << accum_out << std::endl;
  return accum_out;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    view_t view1("view1", DIM);

    for (int i = 0; i < DIM; ++i) {
      view1(i) = i+1;
    }

    hash_view(view1);

    view_t view2("view2", DIM);
    Kokkos::deep_copy(view2, view1);
    hash_view(view2);

    view2(1) = 42;
    hash_view(view2);
  }
  Kokkos::finalize();
  return 0;
}

#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "common.hpp"

using namespace std;

static constexpr int DIM = 3;

using view_t = Kokkos::View<int*>;
using uview_t = Unmanaged<view_t>;

struct ArrayType
{
  int m_data[DIM];

  KOKKOS_INLINE_FUNCTION
  ArrayType() { init(); }

  KOKKOS_INLINE_FUNCTION
  ArrayType(const ArrayType& rhs) {
    for (int i = 0; i < DIM; ++i) m_data[i] = rhs.m_data[i];
  }

  KOKKOS_INLINE_FUNCTION
  ArrayType(const view_t&) { init(); }

  KOKKOS_INLINE_FUNCTION
  void init() {
    for (int i = 0; i < DIM; ++i) m_data[i] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  ArrayType& operator +=(const ArrayType& rhs) {
    for (int i = 0; i < DIM; ++i) m_data[i] += rhs.m_data[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  ArrayType& operator +=(const view_t& rhs) {
    for (int i = 0; i < DIM; ++i) m_data[i] += rhs(i);
    return *this;
  }
};

using aview_t = Kokkos::View<ArrayType*>;
using uaview_t = Unmanaged<aview_t>;

struct SumArray
{
  using reducer = SumArray;
  using value_type = ArrayType;
  using result_view_type = uaview_t;

 private:
  value_type& m_value;

 public:
  KOKKOS_INLINE_FUNCTION
  SumArray(value_type& value) : m_value(value) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const { val.init(); }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return m_value; }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&m_value, 1); }

  KOKKOS_INLINE_FUNCTION
  bool reference_scalar() const { return true; }
};

struct ReduceFunctor
{
  std::vector<view_t> views;

  KOKKOS_INLINE_FUNCTION
  void operator()(size_t i, ArrayType& accum)
  {
    auto curr_view = views[i];
    accum += curr_view;
  }
};

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    view_t view1("view1", DIM);
    view_t view2("view2", DIM);
    view_t view3("view3", DIM);
    view_t accum("accum", DIM);

    for (int i = 0; i < DIM; ++i) {
      view1(i) = i+1;
      view2(i) = i*2;
      view3(i) = i*3;
    }

    ReduceFunctor rf;
    rf.views = {view1, view2, view3};

    ArrayType accum_r = accum;
    Kokkos::parallel_reduce(rf.views.size(), rf, SumArray(accum_r));

    for (int i = 0; i < DIM; ++i) {
      //cout << accum(i) << endl;
      cout << accum_r.m_data[i] << endl;
    }

  }
  Kokkos::finalize();
  return 0;
}

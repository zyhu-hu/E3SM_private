#include "catch2/catch.hpp"

#include "share/scream_types.hpp"
#include "ekat/ekat_pack.hpp"
#include "ekat/kokkos/ekat_kokkos_utils.hpp"
#include "physics/dp/dp_functions.hpp"
#include "physics/dp/dp_functions_f90.hpp"

#include "dp_unit_tests_common.hpp"

namespace scream {
namespace dp {
namespace unit_test {

template <typename D>
struct UnitWrap::UnitTest<D>::TestApplyIopForcing {

  static void run_bfb()
  {
    auto engine = setup_random_test();

    ApplyIopForcingData f90_data[] = {
      // TODO
    };

    static constexpr Int num_runs = sizeof(f90_data) / sizeof(ApplyIopForcingData);

    // Generate random input data
    // Alternatively, you can use the f90_data construtors/initializer lists to hardcode data
    for (auto& d : f90_data) {
      d.randomize(engine);
    }

    // Create copies of data for use by cxx. Needs to happen before fortran calls so that
    // inout data is in original state
    ApplyIopForcingData cxx_data[] = {
      // TODO
    };

    // Assume all data is in C layout

    // Get data from fortran
    for (auto& d : f90_data) {
      // expects data in C layout
      apply_iop_forcing(d);
    }

    // Get data from cxx
    for (auto& d : cxx_data) {
      apply_iop_forcing_f(d.nelemd, d.elem, &d.hvcoord, d.hybrid, d.tl, d.n, d.t_before_advance, d.nets, d.nete);
    }

    // Verify BFB results, all data should be in C layout
    if (SCREAM_BFB_TESTING) {
      for (Int i = 0; i < num_runs; ++i) {
        ApplyIopForcingData& d_f90 = f90_data[i];
        ApplyIopForcingData& d_cxx = cxx_data[i];
        //REQUIRE(d_f90.hvcoord == d_cxx.hvcoord);
      }
    }
  } // run_bfb

};

} // namespace unit_test
} // namespace dp
} // namespace scream

namespace {

TEST_CASE("apply_iop_forcing_bfb", "[dp]")
{
  using TestStruct = scream::dp::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestApplyIopForcing;

  TestStruct::run_bfb();
}

} // empty namespace

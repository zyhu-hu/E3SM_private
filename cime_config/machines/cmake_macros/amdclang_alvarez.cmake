if (COMP_NAME STREQUAL gptl)
  string(APPEND CPPDEFS " -DHAVE_NANOTIME -DBIT64 -DHAVE_SLASHPROC -DHAVE_GETTIMEOFDAY")
endif()
string(APPEND SLIBS " -L$ENV{CRAY_HDF5_PARALLEL_PREFIX}/lib -lhdf5_hl -lhdf5 -L$ENV{CRAY_NETCDF_HDF5PARALLEL_PREFIX} -L$ENV{CRAY_PARALLEL_NETCDF_PREFIX}/lib -lpnetcdf -lnetcdf -lnetcdff")
string(APPEND SLIBS " -lblas -llapack -lamdlibm")
set(CXX_LINKER "FORTRAN")
set(NETCDF_PATH "$ENV{CRAY_NETCDF_HDF5PARALLEL_PREFIX}")
set(NETCDF_C_PATH "$ENV{CRAY_NETCDF_HDF5PARALLEL_PREFIX}")
set(NETCDF_FORTRAN_PATH "$ENV{CRAY_NETCDF_HDF5PARALLEL_PREFIX}")
set(HDF5_PATH "$ENV{CRAY_HDF5_PARALLEL_PREFIX}")
set(PNETCDF_PATH "$ENV{CRAY_PARALLEL_NETCDF_PREFIX}")
if (NOT DEBUG)
  string(APPEND CFLAGS " -O2 -g")
endif()
if (NOT DEBUG)
  string(APPEND CXXFLAGS " -O2 -g")
endif()
if (NOT DEBUG)
  string(APPEND FFLAGS " -O2 -g")
endif()
if (DEBUG)
  string(APPEND FFLAGS " -O0")
  string(APPEND CPPDEFS " -DYAKL_DEBUG")
endif()
#string(APPEND FFLAGS " -march=znver3")
string(APPEND CXX_LIBS " -lstdc++")
set(MPICC "cc")
set(MPICXX "CC")
set(MPIFC "ftn")
set(SCC "clang")
set(SCXX "clang++")
set(SFC "flang")

if (COMP_NAME STREQUAL cism)
  string(APPEND CMAKE_OPTS " -D CISM_GNU=ON")
endif()

set(CXX_LINKER "FORTRAN")
string(APPEND FC_AUTO_R8 " -fdefault-real-8")
string(APPEND FFLAGS " -Mflushz ")
if (compile_threaded)
  string(APPEND FFLAGS " -mp")
endif()

string(APPEND FFLAGS_NOOPT " -O0")
string(APPEND FIXEDFLAGS " -Mfixed")
string(APPEND FREEFLAGS " -Mfreeform")
set(HAS_F2008_CONTIGUOUS "FALSE")
if (compile_threaded)
  string(APPEND LDFLAGS " -mp")
endif()
set(SUPPORTS_CXX "TRUE")

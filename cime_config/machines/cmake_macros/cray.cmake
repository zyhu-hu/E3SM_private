if (compile_threaded)
  string(APPEND CMAKE_C_FLAGS " -h omp")
endif()
string(APPEND CMAKE_C_FLAGS_DEBUG " -g -O0")
string(APPEND CPPDEFS " -DFORTRANUNDERSCORE -DNO_R16 -DCPRCRAY")
string(APPEND CMAKE_Fortran_FLAGS " -f free  -em")
if (compile_threaded)
  string(APPEND CMAKE_Fortran_FLAGS " -h omp")
endif()
if (NOT compile_threaded)
  string(APPEND CMAKE_Fortran_FLAGS " -M1077")
endif()
string(APPEND CMAKE_Fortran_FLAGS_DEBUG " -g -O0")
string(APPEND CMAKE_Fortran_FLAGS_DEBUG " -O0")
set(HAS_F2008_CONTIGUOUS "TRUE")
string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--allow-multiple-definition")
if (compile_threaded)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -h omp")
endif()

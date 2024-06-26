module RtmVar

  use shr_kind_mod , only : r8 => shr_kind_r8
  use shr_const_mod, only : SHR_CONST_CDAY,SHR_CONST_REARTH
  use shr_sys_mod  , only : shr_sys_abort
  use RtmSpmd      , only : masterproc

  implicit none

  ! Constants
  integer, parameter, private ::  iundef = -9999999
  integer, parameter, private ::  rundef = -9999999._r8

  real(r8), public, parameter :: secspday = SHR_CONST_CDAY  ! Seconds per day
  integer,  public, parameter :: isecspday= secspday        ! Integer seconds per day
  real(r8), public, parameter :: spval    = 1.e20_r8        ! special value for real data
  integer , public, parameter :: ispval   = -9999           ! special value for int data
  real(r8) :: re = SHR_CONST_REARTH*0.001_r8                ! radius of earth (km)
  logical , public :: barrier_timers                        ! barrier timers

  ! Run control variables
  character(len=256), public :: caseid  = ' '          ! case id
  character(len=256), public :: ctitle  = ' '          ! case title
  integer, public, parameter :: nsrStartup  = 0        ! Startup from initial conditions
  integer, public, parameter :: nsrContinue = 1        ! Continue from restart files
  integer, public, parameter :: nsrBranch   = 2        ! Branch from restart files
  integer, public :: nsrest = iundef                   ! Type of run
  logical, public :: brnch_retain_casename = .false.   ! true => allow case name to remain the same for branch run
                                                       ! by default this is not allowed
  logical, public :: wrmflag = .false.                 ! water management model flag
  logical, public :: inundflag = .false.               ! inundation model flag
  logical, public :: use_lnd_rof_two_way = .false.     ! land river two way coupling flag
  logical, public :: sediflag = .false.                ! sediment model flag
  logical, public :: heatflag = .false.                ! heat model flag
  logical, public :: use_ocn_rof_two_way = .false.     ! ocean river two way coupling flag
  logical, public :: rstraflag = .false.               ! reservoir stratification module flag
  real,    public :: rinittemp = 283.15_r8             ! initial reservoir temperature
  integer, public :: ngeom   = 50                      ! Reservoir depth layers to calculate depth-area-storage relationship
  integer, public :: nlayers = 30                      ! Maximum number of reservoir layers for stratification
  logical, public :: noland = .false.                  ! true => no valid land points -- do NOT run
  logical, public :: data_bgc_fluxes_to_ocean_flag = .false.! read in and send BGC fluxes to ocean flag
  character(len=32) , public :: decomp_option          ! decomp option
  character(len=32) , public :: smat_option            ! smatrix multiply option (opt, Xonly, Yonly)
                                                       ! opt   = XandY in MCT
                                                       ! Xonly = Xonly in MCT, should be bfb on different pe counts
                                                       ! Yonly = Yonly in MCT
  character(len=256), public :: hostname = ' '         ! Hostname of machine running on
  character(len=256), public :: username = ' '         ! username of user running program
  character(len=256), public :: version  = " "         ! version of program
  character(len=256), public :: conventions = "CF-1.7" ! dataset conventions
  character(len=256), public :: source   = "MOSART in E3SM" ! description of this source

  ! Unit Numbers
  integer, public :: iulog = 6        ! "stdout" log file unit number, default is 6

  ! Instance control
  integer, public :: inst_index
  character(len=16), public :: inst_name
  character(len=16), public :: inst_suffix

  ! Rtm control variables
  character(len=256), public :: nrevsn_rtm   = ' '   ! restart data file name for branch run
  character(len=256), public :: finidat_rtm  = ' '   ! initial conditions file name
  character(len=256), public :: frivinp_rtm  = ' '   ! RTM input data file name
  character(len=256), public :: frivinp_mesh  = ' '   ! mesh input data file name
  logical,            public :: ice_runoff = .true.  ! true => runoff is split into liquid and ice, 
                                                     ! otherwise just liquid
  ! Rtm grid size
  integer :: rtmlon = 1 ! number of rtm longitudes (initialize)
  integer :: rtmlat = 1 ! number of rtm latitudes  (initialize)
  logical :: isgrid2d = .true. ! Determine if the inputs are 1d or 2d

  character(len=256), public :: rpntfil = 'rpointer.rof' ! file name for local restart pointer file

  logical, private :: RtmVar_isset = .false.

contains


!================================================================================

  subroutine RtmVarSet( caseid_in, ctitle_in, brnch_retain_casename_in,    &
                        nsrest_in, version_in, hostname_in, username_in )

    !-----------------------------------------------------------------------
    !  Set input control variables.
    !
    ! !ARGUMENTS:
    character(len=256), optional, intent(IN) :: caseid_in    ! case id
    character(len=256), optional, intent(IN) :: ctitle_in    ! case title
    integer,            optional, intent(IN) :: nsrest_in    ! 0: initial run. 1: restart: 3: branch
    character(len=256), optional, intent(IN) :: version_in   ! model version
    character(len=256), optional, intent(IN) :: hostname_in  ! hostname running on
    character(len=256), optional, intent(IN) :: username_in  ! username running job
    logical,            optional, intent(IN) :: brnch_retain_casename_in ! true => allow case name to
    !-----------------------------------------------------------------------

    if ( RtmVar_isset )then
       call shr_sys_abort( 'RtmVarSet ERROR:: control variables already set -- EXIT' )
    end if

    if (present(caseid_in)) caseid = caseid_in
    if (present(ctitle_in)) ctitle = ctitle_in
    if (present(nsrest_in)) nsrest = nsrest_in
    if (present(version_in)) version = version_in
    if (present(username_in)) username = username_in
    if (present(hostname_in)) hostname = hostname_in
    if (present(brnch_retain_casename_in)) brnch_retain_casename = brnch_retain_casename_in

  end subroutine RtmVarSet

!================================================================================

  subroutine RtmVarInit( )
    if (masterproc) then
       if (nsrest == iundef) then
          call shr_sys_abort( 'RtmVarInit ERROR:: must set nsrest' )
       end if
       if (nsrest == nsrBranch .and. nrevsn_rtm == ' ') then
          call shr_sys_abort( 'RtmVarInit ERROR: need to set restart data file name' )
       end if
       if (nsrest == nsrStartup ) then
          nrevsn_rtm = ' '
       end if
       if (nsrest == nsrContinue) then
          nrevsn_rtm = 'set by restart pointer file file'
       end if
       if (nsrest /= nsrStartup .and. nsrest /= nsrContinue .and. nsrest /= nsrBranch ) then
          call shr_sys_abort( 'RtmVarInit ERROR: nsrest NOT set to a valid value' )
       end if
    endif   
    RtmVar_isset = .true.
  end subroutine RtmVarInit

end module RtmVar

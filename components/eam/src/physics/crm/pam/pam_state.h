#pragma once

#include "pam_coupler.h"

// wrapper for PAM's set_grid
inline void pam_state_set_grid( pam::PamCoupler &coupler ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  auto &dm_device = coupler.get_data_manager_device_readwrite();
  auto &dm_host   = coupler.get_data_manager_host_readwrite();
  auto nens       = coupler.get_option<int>("ncrms");
  auto crm_nz     = coupler.get_option<int>("crm_nz");
  auto gcm_nlev   = coupler.get_option<int>("gcm_nlev");
  auto crm_dx     = coupler.get_option<double>("crm_dx");
  auto crm_dy     = coupler.get_option<double>("crm_dy");
  //------------------------------------------------------------------------------------------------
  // Set the vertical grid in the coupler (need to flip the vertical dimension of input data)
  auto input_zint = dm_host.get<real const,2>("input_zint").createDeviceCopy();
  auto input_phis = dm_host.get<real const,1>("input_phis").createDeviceCopy();
  real2d zint_tmp("zint_tmp",crm_nz+1,nens);
  // auto grav = coupler.get_option<double>("grav");
  real grav = 9.80616; // note - we can't use the coupler grav because it hasn't been set yet
  parallel_for( Bounds<2>(crm_nz+1,nens) , YAKL_LAMBDA (int k_crm, int iens) {
    int k_gcm = (gcm_nlev+1)-1-k_crm;
    zint_tmp(k_crm,iens) = input_zint(k_gcm,iens) + input_phis(iens)/grav;
  });
  coupler.set_grid( crm_dx , crm_dy , zint_tmp );
  //------------------------------------------------------------------------------------------------
}


// update the coupler GCM state variables using the input GCM state
inline void pam_state_update_gcm_state( pam::PamCoupler &coupler ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  auto &dm_device = coupler.get_data_manager_device_readwrite();
  auto &dm_host   = coupler.get_data_manager_host_readwrite();
  auto nens       = coupler.get_option<int>("ncrms");
  auto crm_nz     = coupler.get_option<int>("crm_nz");
  auto gcm_nlev   = coupler.get_option<int>("gcm_nlev");
  real R_d        = coupler.get_option<real>("R_d");
  real R_v        = coupler.get_option<real>("R_v");
  real cp_d       = coupler.get_option<real>("cp_d");
  real grav       = coupler.get_option<real>("grav");
  real Lv         = coupler.get_option<real>("latvap") ;
  real Lf         = coupler.get_option<real>("latice") ;
  //------------------------------------------------------------------------------------------------
  // get the coupler GCM state arrays used to force the CRM
  auto gcm_rho_d = dm_device.get<real,2>("gcm_density_dry");
  auto gcm_uvel  = dm_device.get<real,2>("gcm_uvel"       );
  auto gcm_vvel  = dm_device.get<real,2>("gcm_vvel"       );
  auto gcm_temp  = dm_device.get<real,2>("gcm_temp"       );
  auto gcm_rho_v = dm_device.get<real,2>("gcm_water_vapor");
  auto gcm_rho_l = dm_device.get<real,2>("gcm_cloud_water");
  auto gcm_rho_i = dm_device.get<real,2>("gcm_cloud_ice"  );
  //------------------------------------------------------------------------------------------------
  // wrap the host GCM state data in YAKL arrays
  auto input_ul   = dm_host.get<real const,2>("input_ul"  ).createDeviceCopy();
  auto input_vl   = dm_host.get<real const,2>("input_vl"  ).createDeviceCopy();
  auto input_tl   = dm_host.get<real const,2>("input_tl"  ).createDeviceCopy();
  auto input_qccl = dm_host.get<real const,2>("input_qccl").createDeviceCopy();
  auto input_qiil = dm_host.get<real const,2>("input_qiil").createDeviceCopy();
  auto input_ql   = dm_host.get<real const,2>("input_ql"  ).createDeviceCopy();
  auto input_pmid = dm_host.get<real const,2>("input_pmid").createDeviceCopy();
  auto input_pint = dm_host.get<real const,2>("input_pint").createDeviceCopy();
  auto input_pdel = dm_host.get<real const,2>("input_pdel").createDeviceCopy();
  auto input_zint = dm_host.get<real const,2>("input_zint").createDeviceCopy();
  //------------------------------------------------------------------------------------------------
  auto state_rho_dry = dm_host.get<real const,4>("state_rho_dry").createDeviceCopy();
  // Define GCM state for forcing - adjusted to avoid directly forcing cloud liquid and ice fields
  parallel_for( Bounds<2>(crm_nz,nens) , YAKL_LAMBDA (int k_crm, int iens) {
    int k_gcm = gcm_nlev-1-k_crm;

    gcm_uvel (k_crm,iens) = input_ul(k_gcm,iens);
    gcm_vvel (k_crm,iens) = input_vl(k_gcm,iens);

    // use total water for GCM forcing
    real input_qt = input_ql(k_gcm,iens) + input_qccl(k_gcm,iens) + input_qiil(k_gcm,iens);

    // calculate dry density using same formula as in crm_physics_tend()
    real dz = input_zint(k_gcm,iens) - input_zint(k_gcm+1,iens);
    real dp = input_pint(k_gcm,iens) - input_pint(k_gcm+1,iens);
    #ifdef MMF_PAM_DYCOR_AWFL
      // when forcing dry density use total water for consistency with temperature forcing
      gcm_rho_d(k_crm,iens) = -1 * dp * (1-input_ql(k_gcm,iens)) / ( dz * grav );
    #endif
    #ifdef MMF_PAM_DYCOR_SPAM
      gcm_rho_d(k_crm,iens) = -1 * dp * (1-input_ql(k_gcm,iens)) / ( dz * grav );
    #endif

    // convert total water mixing ratio to water vapor density and adjust temperature
    gcm_rho_v(k_crm,iens) = input_qt * gcm_rho_d(k_crm,iens) / ( 1 - input_qt );
    gcm_rho_l(k_crm,iens) = 0;
    gcm_rho_i(k_crm,iens) = 0;
    real input_t_adj = input_tl(k_gcm,iens) - ( input_qccl(k_gcm,iens)*Lv + input_qiil(k_gcm,iens)*(Lv+Lf) ) / cp_d ;
    gcm_temp(k_crm,iens) = input_t_adj;

    // // Alternate version where we ignore liq/ice cloud in the large-scale forcing
    // gcm_rho_v(k_crm,iens) = input_ql(k_gcm,iens) * gcm_rho_d(k_crm,iens) / ( 1 - input_ql(k_gcm,iens) );
    // gcm_temp(k_crm,iens) = input_tl(k_gcm,iens);

    // // another alternate where we force cloud water/ice separately
    // gcm_rho_v(k_crm,iens) = input_ql  (k_gcm,iens) * gcm_rho_d(k_crm,iens) / ( 1 - input_ql(k_gcm,iens) );
    // gcm_rho_l(k_crm,iens) = input_qccl(k_gcm,iens) * ( gcm_rho_d(k_crm,iens) + gcm_rho_v(k_crm,iens) );
    // gcm_rho_i(k_crm,iens) = input_qiil(k_gcm,iens) * ( gcm_rho_d(k_crm,iens) + gcm_rho_v(k_crm,iens) );
    // gcm_temp(k_crm,iens) = input_tl(k_gcm,iens);
  });

  //------------------------------------------------------------------------------------------------
}

// update CRM dry density
inline void pam_state_update_dry_density( pam::PamCoupler &coupler ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  auto &dm_device = coupler.get_data_manager_device_readwrite();
  auto &dm_host   = coupler.get_data_manager_host_readwrite();
  auto nens       = coupler.get_option<int>("ncrms");
  auto nz         = coupler.get_option<int>("crm_nz");
  auto nx         = coupler.get_option<int>("crm_nx");
  auto ny         = coupler.get_option<int>("crm_ny");
  auto crm_rho_d  = dm_device.get<real,4>("density_dry");
  //------------------------------------------------------------------------------------------------
  #ifdef MMF_PAM_DYCOR_AWFL
    // When forcing dry density we need to restore the previous CRM field, similar to other state variables
    auto state_rho_dry       = dm_host.get<real const,4>("state_rho_dry").createDeviceCopy();
    parallel_for("Copy in CRM state dry density", SimpleBounds<4>(nz,ny,nx,nens), YAKL_LAMBDA (int k, int j, int i, int iens) {
      crm_rho_d(k,j,i,iens) = state_rho_dry(k,j,i,iens);
    });
    // std::cout<<"WHDEBUG - ERROR - still using AWFL?!"<<std::endl;
  #endif
  #ifdef MMF_PAM_DYCOR_SPAM
    // make sure CRM density matches GCM so that dryd ensity forcing is zero
    auto gcm_rho_d = dm_device.get<real,2>("gcm_density_dry");
    parallel_for("Broadcast GCM dry density", SimpleBounds<4>(nz,ny,nx,nens), YAKL_LAMBDA (int k, int j, int i, int iens) {
      crm_rho_d(k,j,i,iens) = gcm_rho_d(k,iens);
    });
  #endif
  //------------------------------------------------------------------------------------------------
}


// Copy the CRM state saved by the GCM into the PAM coupler
inline void pam_state_copy_input_to_coupler( pam::PamCoupler &coupler ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  auto &dm_device = coupler.get_data_manager_device_readwrite();
  auto &dm_host   = coupler.get_data_manager_host_readwrite();
  auto nens       = coupler.get_option<int>("ncrms");
  auto nz         = coupler.get_option<int>("crm_nz");
  auto nx         = coupler.get_option<int>("crm_nx");
  auto ny         = coupler.get_option<int>("crm_ny");
  //------------------------------------------------------------------------------------------------
  // get the coupler state variables
  auto crm_rho_d         = dm_device.get<real,4>("density_dry");
  auto crm_uvel          = dm_device.get<real,4>("uvel");
  auto crm_vvel          = dm_device.get<real,4>("vvel");
  auto crm_wvel          = dm_device.get<real,4>("wvel");
  auto crm_temp          = dm_device.get<real,4>("temp");
  auto crm_rho_v         = dm_device.get<real,4>("water_vapor");
  auto crm_rho_c         = dm_device.get<real,4>("cloud_water");
  auto crm_rho_r         = dm_device.get<real,4>("rain");
  auto crm_rho_i         = dm_device.get<real,4>("ice");
  auto crm_nc            = dm_device.get<real,4>("cloud_water_num");
  auto crm_nr            = dm_device.get<real,4>("rain_num");
  auto crm_ni            = dm_device.get<real,4>("ice_num");
  auto crm_qm            = dm_device.get<real,4>("ice_rime");
  auto crm_bm            = dm_device.get<real,4>("ice_rime_vol");
  auto crm_t_prev        = dm_device.get<real,4>("qv_prev");
  auto crm_q_prev        = dm_device.get<real,4>("t_prev");
  auto crm_shoc_tk       = dm_device.get<real,4>("tk");
  auto crm_shoc_tkh      = dm_device.get<real,4>("tkh");
  auto crm_shoc_wthv     = dm_device.get<real,4>("wthv_sec");
  auto crm_shoc_relvar   = dm_device.get<real,4>("relvar");
  auto crm_shoc_cldfrac  = dm_device.get<real,4>("cldfrac");
  //------------------------------------------------------------------------------------------------
  // wrap the host CRM state data in YAKL arrays
  auto state_u_wind        = dm_host.get<real const,4>("state_u_wind").createDeviceCopy();
  auto state_v_wind        = dm_host.get<real const,4>("state_v_wind").createDeviceCopy();
  auto state_w_wind        = dm_host.get<real const,4>("state_w_wind").createDeviceCopy();
  auto state_temperature   = dm_host.get<real const,4>("state_temperature").createDeviceCopy();
  auto state_qv            = dm_host.get<real const,4>("state_qv").createDeviceCopy();
  auto state_qc            = dm_host.get<real const,4>("state_qc").createDeviceCopy();
  auto state_qr            = dm_host.get<real const,4>("state_qr").createDeviceCopy();
  auto state_qi            = dm_host.get<real const,4>("state_qi").createDeviceCopy();
  auto state_nc            = dm_host.get<real const,4>("state_nc").createDeviceCopy();
  auto state_nr            = dm_host.get<real const,4>("state_nr").createDeviceCopy();
  auto state_ni            = dm_host.get<real const,4>("state_ni").createDeviceCopy();
  auto state_qm            = dm_host.get<real const,4>("state_qm").createDeviceCopy();
  auto state_bm            = dm_host.get<real const,4>("state_bm").createDeviceCopy();
  auto state_t_prev        = dm_host.get<real const,4>("state_t_prev").createDeviceCopy();
  auto state_q_prev        = dm_host.get<real const,4>("state_q_prev").createDeviceCopy();
  auto state_shoc_tk       = dm_host.get<real const,4>("state_shoc_tk").createDeviceCopy();
  auto state_shoc_tkh      = dm_host.get<real const,4>("state_shoc_tkh").createDeviceCopy();
  auto state_shoc_wthv     = dm_host.get<real const,4>("state_shoc_wthv").createDeviceCopy();
  auto state_shoc_relvar   = dm_host.get<real const,4>("state_shoc_relvar").createDeviceCopy();
  auto state_shoc_cldfrac  = dm_host.get<real const,4>("state_shoc_cldfrac").createDeviceCopy();
  //------------------------------------------------------------------------------------------------
  // Copy the host CRM data to the coupler
  parallel_for("Horz mean of CRM state", SimpleBounds<4>(nz,ny,nx,nens), YAKL_LAMBDA (int k, int j, int i, int iens) {
    // NOTE - convert specific mass mixing ratios to density
    crm_rho_v        (k,j,i,iens) = state_qv(k,j,i,iens) * crm_rho_d(k,j,i,iens) / ( 1 - state_qv(k,j,i,iens) ) ;
    crm_rho_c        (k,j,i,iens) = state_qc(k,j,i,iens) * ( crm_rho_d(k,j,i,iens) + crm_rho_v(k,j,i,iens) ) ;
    crm_rho_r        (k,j,i,iens) = state_qr(k,j,i,iens) * ( crm_rho_d(k,j,i,iens) + crm_rho_v(k,j,i,iens) ) ;
    crm_rho_i        (k,j,i,iens) = state_qi(k,j,i,iens) * ( crm_rho_d(k,j,i,iens) + crm_rho_v(k,j,i,iens) ) ;
    crm_uvel         (k,j,i,iens) = state_u_wind       (k,j,i,iens);
    crm_vvel         (k,j,i,iens) = state_v_wind       (k,j,i,iens);
    crm_wvel         (k,j,i,iens) = state_w_wind       (k,j,i,iens);
    crm_temp         (k,j,i,iens) = state_temperature  (k,j,i,iens);
    crm_nc           (k,j,i,iens) = state_nc           (k,j,i,iens);
    crm_nr           (k,j,i,iens) = state_nr           (k,j,i,iens);
    crm_ni           (k,j,i,iens) = state_ni           (k,j,i,iens);
    crm_qm           (k,j,i,iens) = state_qm           (k,j,i,iens);
    crm_bm           (k,j,i,iens) = state_bm           (k,j,i,iens);
    crm_t_prev       (k,j,i,iens) = state_t_prev       (k,j,i,iens);
    crm_q_prev       (k,j,i,iens) = state_q_prev       (k,j,i,iens);
    crm_shoc_tk      (k,j,i,iens) = state_shoc_tk      (k,j,i,iens);
    crm_shoc_tkh     (k,j,i,iens) = state_shoc_tkh     (k,j,i,iens);
    crm_shoc_wthv    (k,j,i,iens) = state_shoc_wthv    (k,j,i,iens);
    crm_shoc_relvar  (k,j,i,iens) = state_shoc_relvar  (k,j,i,iens);
    crm_shoc_cldfrac (k,j,i,iens) = state_shoc_cldfrac (k,j,i,iens);
  });
  //------------------------------------------------------------------------------------------------
}

// 
inline void pam_state_copy_to_host( pam::PamCoupler &coupler ) {
  using yakl::c::parallel_for;
  using yakl::c::SimpleBounds;
  using yakl::atomicAdd;
  auto &dm_device = coupler.get_data_manager_device_readwrite();
  auto &dm_host   = coupler.get_data_manager_host_readwrite();
  int nz   = dm_device.get_dimension_size("z"   );
  int ny   = dm_device.get_dimension_size("y"   );
  int nx   = dm_device.get_dimension_size("x"   );
  int nens = dm_device.get_dimension_size("nens");
  //------------------------------------------------------------------------------------------------
  auto crm_rho_d                = dm_device.get<real,4>("density_dry");
  auto crm_uvel                 = dm_device.get<real,4>("uvel");
  auto crm_vvel                 = dm_device.get<real,4>("vvel");
  auto crm_wvel                 = dm_device.get<real,4>("wvel");
  auto crm_temp                 = dm_device.get<real,4>("temp");
  auto crm_rho_v                = dm_device.get<real,4>("water_vapor");
  auto crm_rho_c                = dm_device.get<real,4>("cloud_water");
  auto crm_rho_r                = dm_device.get<real,4>("rain");
  auto crm_rho_i                = dm_device.get<real,4>("ice");
  auto crm_num_c                = dm_device.get<real,4>("cloud_water_num");
  auto crm_num_r                = dm_device.get<real,4>("rain_num");
  auto crm_num_i                = dm_device.get<real,4>("ice_num");
  auto crm_qm                   = dm_device.get<real,4>("ice_rime");
  auto crm_bm                   = dm_device.get<real,4>("ice_rime_vol");
  auto crm_t_prev               = dm_device.get<real,4>("qv_prev");
  auto crm_q_prev               = dm_device.get<real,4>("t_prev");
  auto crm_shoc_tk              = dm_device.get<real,4>("tk");
  auto crm_shoc_tkh             = dm_device.get<real,4>("tkh");
  auto crm_shoc_wthv            = dm_device.get<real,4>("wthv_sec");
  auto crm_shoc_relvar          = dm_device.get<real,4>("relvar");
  auto crm_shoc_cldfrac         = dm_device.get<real,4>("cldfrac");
  //------------------------------------------------------------------------------------------------
  // wrap the host CRM state data in YAKL arrays
  auto host_state_u_wind        = dm_host.get<real,4>("state_u_wind");
  auto host_state_v_wind        = dm_host.get<real,4>("state_v_wind");
  auto host_state_w_wind        = dm_host.get<real,4>("state_w_wind");
  auto host_state_temperature   = dm_host.get<real,4>("state_temperature");
  auto host_state_rho_dry       = dm_host.get<real,4>("state_rho_dry");
  auto host_state_qv            = dm_host.get<real,4>("state_qv");
  auto host_state_qc            = dm_host.get<real,4>("state_qc");
  auto host_state_qr            = dm_host.get<real,4>("state_qr");
  auto host_state_qi            = dm_host.get<real,4>("state_qi");
  auto host_state_nc            = dm_host.get<real,4>("state_nc");
  auto host_state_nr            = dm_host.get<real,4>("state_nr");
  auto host_state_ni            = dm_host.get<real,4>("state_ni");
  auto host_state_qm            = dm_host.get<real,4>("state_qm");
  auto host_state_bm            = dm_host.get<real,4>("state_bm");
  auto host_state_t_prev        = dm_host.get<real,4>("state_t_prev");
  auto host_state_q_prev        = dm_host.get<real,4>("state_q_prev");
  auto host_state_shoc_tk       = dm_host.get<real,4>("state_shoc_tk");
  auto host_state_shoc_tkh      = dm_host.get<real,4>("state_shoc_tkh");
  auto host_state_shoc_wthv     = dm_host.get<real,4>("state_shoc_wthv");
  auto host_state_shoc_relvar   = dm_host.get<real,4>("state_shoc_relvar");
  auto host_state_shoc_cldfrac  = dm_host.get<real,4>("state_shoc_cldfrac");
  //------------------------------------------------------------------------------------------------
  // convert densities back to specific mixing ratios
  real4d tmp_qv("tmp_qv",nz,ny,nx,nens);
  real4d tmp_qc("tmp_qc",nz,ny,nx,nens);
  real4d tmp_qr("tmp_qr",nz,ny,nx,nens);
  real4d tmp_qi("tmp_qi",nz,ny,nx,nens);
  parallel_for("Horz mean of CRM state", SimpleBounds<4>(nz,ny,nx,nens), YAKL_LAMBDA (int k, int j, int i, int iens) {
    // convert density to specific mixing ratio
    real tmp_rho = crm_rho_d(k,j,i,iens)+crm_rho_v(k,j,i,iens);
    tmp_qv(k,j,i,iens) = crm_rho_v(k,j,i,iens) / tmp_rho;
    tmp_qc(k,j,i,iens) = crm_rho_c(k,j,i,iens) / tmp_rho;
    tmp_qr(k,j,i,iens) = crm_rho_r(k,j,i,iens) / tmp_rho;
    tmp_qi(k,j,i,iens) = crm_rho_i(k,j,i,iens) / tmp_rho;
  });
  //------------------------------------------------------------------------------------------------
  // Copy the CRM state to host arrays
  crm_uvel          .deep_copy_to( host_state_u_wind        );
  crm_vvel          .deep_copy_to( host_state_v_wind        );
  crm_wvel          .deep_copy_to( host_state_w_wind        );
  crm_temp          .deep_copy_to( host_state_temperature   );
  crm_rho_d         .deep_copy_to( host_state_rho_dry       );
  tmp_qv            .deep_copy_to( host_state_qv            );
  tmp_qc            .deep_copy_to( host_state_qc            );
  tmp_qi            .deep_copy_to( host_state_qi            );
  tmp_qr            .deep_copy_to( host_state_qr            );
  crm_num_c         .deep_copy_to( host_state_nc            );
  crm_num_r         .deep_copy_to( host_state_nr            );
  crm_num_i         .deep_copy_to( host_state_ni            );
  crm_qm            .deep_copy_to( host_state_qm            );
  crm_bm            .deep_copy_to( host_state_bm            );
  crm_t_prev        .deep_copy_to( host_state_t_prev        );
  crm_q_prev        .deep_copy_to( host_state_q_prev        );
  crm_shoc_tk       .deep_copy_to( host_state_shoc_tk       );
  crm_shoc_tkh      .deep_copy_to( host_state_shoc_tkh      );
  crm_shoc_wthv     .deep_copy_to( host_state_shoc_wthv     );
  crm_shoc_relvar   .deep_copy_to( host_state_shoc_relvar   );
  crm_shoc_cldfrac  .deep_copy_to( host_state_shoc_cldfrac  );
  //------------------------------------------------------------------------------------------------
}

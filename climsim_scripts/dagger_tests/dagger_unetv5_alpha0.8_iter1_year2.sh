#!/bin/bash
set -e

exp_vars="cb_partial_coupling_vars = 'ptend_t', 'ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out_PRECC', 'cam_out_PRECSC', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'"

exp_cb_ramp_option="cb_ramp_option='constant'"
exp_cb_ramp_factor="cb_ramp_factor = 0.8"
exp_cb_do_ramp="cb_do_ramp = .true."
exp_cb_partial_coupling = "cb_partial_coupling = .true."

dir_ens_root=$(pwd)

export SBATCH_ACCOUNT=m4331

for i in $(seq 1 12); do
    refdate="0002-$(printf "%02d" $i)-02"
    uniqueid="ens-${refdate}"
    # reftod_values=(00000 00000 00000 00000)
    # reftod=${reftod_values[$(((i - 1) % 4))]}
    reftod="00000"
    parent_name="dagger_unetv5_alpha0.8_iter1"
    expnamenew="${uniqueid}"

    # create clone
    dir_case_orig=/global/homes/z/zeyuanhu/scratch/e3sm_mlt_scratch/seed_partial_v5_classifier_qnlog_0tropopause_reverse_dtheta/
    dir_case_new=${dir_ens_root}/${parent_name}/$expnamenew
    /global/homes/z/zeyuanhu/nvidia_codes/E3SM_private/cime/scripts/create_clone --clone ${dir_case_orig} \
                                                                                  --case ${dir_case_new} \
                                                                                  --keepexe

    # move to ens dir
    cd ${dir_case_new}

    # make modifications of case files
    mkdir run
    mkdir archive

    ./xmlchange BUILD_COMPLETE=TRUE
    ./xmlchange RUNDIR=$(pwd)/run
    ./xmlchange DOUT_S_ROOT=$(pwd)/archive
    ./xmlchange DOUT_S=FALSE

    ./xmlchange JOB_QUEUE=regular
    ./xmlchange STOP_N=30
    ./xmlchange STOP_OPTION='ndays'
    ./xmlchange RESUBMIT=0
    ./xmlchange JOB_WALLCLOCK_TIME="02:00:00"
    ./xmlchange RUN_REFDIR="/pscratch/sd/z/zeyuanhu/e3sm_mlt_scratch/E3SM_ML_ne4_rerun.F2010-MMF1/archive/rest/${refdate}-${reftod}"
    ./xmlchange RUN_REFDATE="${refdate}"
    ./xmlchange RUN_REFTOD="${reftod}"
    # modify atm_in
    ./preview_namelists

    # overwrite
    sed -i "s|.*cb_partial_coupling_vars.*|${exp_vars}|g" ./user_nl_eam
    sed -i "s|.*cb_do_ramp.*|${exp_cb_do_ramp}|g" ./user_nl_eam
    sed -i "s|.*cb_ramp_factor.*|${exp_cb_ramp_factor}|g" ./user_nl_eam
    sed -i "s|.*cb_ramp_option.*|${exp_cb_ramp_option}|g" ./user_nl_eam
    sed -i "s|.*cb_partial_coupling.*|${exp_cb_partial_coupling}|g" ./user_nl_eam
    # submit
    echo -e "\033[0;32m $ID \033[0m"
    ./case.submit

    # move back to the original directory
    cd ${dir_ens_root}
done
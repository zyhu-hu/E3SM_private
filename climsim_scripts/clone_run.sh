#!/bin/bash
set -e

# exp_type=ALL
exp_vars="cb_partial_coupling_vars = 'ptend_t', 'ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out_PRECC', 'cam_out_PRECSC', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'"

dir_ens_root=`pwd`

export SBATCH_ACCOUNT=m4331

# for kens in `seq -w 036 160`; do

# dir_subfolders=/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/v3_hpo_weighthpo

# Changed from a sequence to iterating over subfolder names
# for subfolder in $(find ${dir_subfolders} -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do
#   kens=$(basename ${subfolder})
#   echo "Processing ${kens}"
ID="v3_mlp_wmask_cdice_q1em8_rop2_lr1em3_flip_rest2"
# create clone
dir_case_orig=/global/homes/z/zeyuanhu/scratch/e3sm_mlt_scratch/torch-mlp-v3-qprune-allvar-template
dir_case_new=${dir_ens_root}/$ID
/global/homes/z/zeyuanhu/nvidia_codes/E3SM_private/cime/scripts/create_clone --clone ${dir_case_orig}\
                                                                              --case ${dir_case_new}\
                                                                              --keepexe

# move to ens dir
cd ${dir_case_new}

# make modifications of case files
mkdir run
mkdir archive

./xmlchange BUILD_COMPLETE=TRUE
./xmlchange RUNDIR=`pwd`/run
./xmlchange DOUT_S_ROOT=`pwd`/archive

./xmlchange JOB_QUEUE=regular
./xmlchange STOP_N=15
./xmlchange STOP_OPTION='ndays'
./xmlchange JOB_WALLCLOCK_TIME="00:30:00"
# ./xmlchange CHARGE_ACCOUNT={acct},PROJECT={acct}
# ./xmlchange STOP_OPTION={stop_opt},STOP_N={stop_n},RESUBMIT={resub}'

# modify atm_in
./preview_namelists
# overwrite fkb weight file
f_torch_wgt=`echo /global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/${ID}/model.pt`
f_inp_sub=`echo /global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/${ID}/inp_sub.txt`
f_inp_div=`echo /global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/${ID}/inp_div.txt`
f_out_scale=`echo /global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/${ID}/out_scale.txt`
sed -i "s|.*cb_torch_model.*|cb_torch_model = '${f_torch_wgt}'|g" ./user_nl_eam
sed -i "s|.*cb_inp_sub.*|cb_inp_sub = '${f_inp_sub}'|g" ./user_nl_eam
sed -i "s|.*cb_inp_div.*|cb_inp_div = '${f_inp_div}'|g" ./user_nl_eam
sed -i "s|.*cb_out_scale.*|cb_out_scale = '${f_out_scale}'|g" ./user_nl_eam
# overwrite
sed -i "s|.*cb_partial_coupling_vars.*|${exp_vars}|g" ./user_nl_eam

# submit
echo -e "\033[0;32m $ID \033[0m"
./case.submit
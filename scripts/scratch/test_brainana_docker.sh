fs_license=/mnt/DataDrive3/xliu/prep_test/freesurfer_license.txt
version=2.1.1

# # ------------------------------------------------------------
# # # 1. devtest
bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_devtest
output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_devtest_docker_v${version}
config_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_res-1.yaml

# # ------------------------------------------------------------
# # # # newcastle - sub-03
# bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_newcastle
# output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_newcastle
# config_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_res-1.yaml

# run docker without custom template
docker run --rm -t --gpus all \
    -v "$bids_dir":/input \
    -v "$output_dir":/output \
    -v "$fs_license":/fs_license.txt \
    -v "$config_f":/config.yaml \
    brainana:latest \
    /input /output/preprocessed \
    -w /output/preprocessed_wd \
    --config /config.yaml \
    --freesurfer-license /fs_license.txt 

# ------------------------------------------------------------
# # # 2. prime-de
# # site=site-amu      
# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE/${site}
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/${site}

# # # ------------------------------------------------------------
# # # 3. sub-example
# bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_example
# output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_example

# # run docker without custom template
# docker run --rm -t --gpus all \
#     -v "$bids_dir":/input \
#     -v "$output_dir":/output \
#     -v "$fs_license":/fs_license.txt \
#     brainana:latest \
#     /input /output/preprocessed \
#     -w /output/preprocessed_wd \
#     --freesurfer-license /fs_license.txt 

# # ------------------------------------------------------------
# # with custom template
# custom_template_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/tpl-MEBRAINS_res-1_T1w_brain.nii.gz
# output_dir=${output_dir}_customtemplate

# docker run --rm -t --gpus all \
#     -v "$bids_dir":/input \
#     -v "$output_dir":/output \
#     -v "$fs_license":/fs_license.txt \
#     -v "$config_f":/config.yaml \
#     -v "$custom_template_f":/custom_template.nii.gz \
#     brainana:latest \
#     /input /output/preprocessed \
#     -w /output/preprocessed_wd \
#     --config /config.yaml \
#     --freesurfer-license /fs_license.txt \
#     --output_space /custom_template.nii.gz

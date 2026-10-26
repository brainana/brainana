fs_license=/mnt/DataDrive3/xliu/prep_test/freesurfer_license.txt
version=3.3.0

custom_template_f=''
config_f=''

# # ------------------------------------------------------------
# # # 1. devtest
bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_devtest
output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_devtest_docker_v${version}_sub
config_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_res-1.yaml

# ------------------------------------------------------------
# # # 2. prime-de
# # site=site-amu      
# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE/${site}
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/${site}

# # # ------------------------------------------------------------
# # # 3. sub-example
# bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_example
# output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_example_docker_v${version}

# # ------------------------------------------------------------
# # # # 4. UNC-Wisconsin
# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/UNC-Wisconsin/bids
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/UNC-Wisconsin_brainana_v${version}
# config_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_ses.yaml

# # ------------------------------------------------------------
# # klink
# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/chris_klink/bids
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/klink_brainana_v${version}

# # ------------------------------------------------------------
# # with custom template
# output_dir=${output_dir}_customtemplate
# custom_template_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/tpl-MEBRAINS_res-1_T1w_brain.nii.gz

# ------
# run docker (optional --config / --output_space when paths are set)
DOCKER_VOLUMES=(
    -v "$bids_dir":/input
    -v "$output_dir":/output
    -v "$fs_license":/fs_license.txt
)
DOCKER_ARGS=(
    /input /output/preprocessed
    -w /output/preprocessed_wd
    --freesurfer-license /fs_license.txt
)
if [ -n "$config_f" ]; then
    DOCKER_VOLUMES+=(-v "$config_f":/config.yaml)
    DOCKER_ARGS+=(--config /config.yaml)
fi
if [ -n "$custom_template_f" ]; then
    DOCKER_VOLUMES+=(-v "$custom_template_f":/custom_template.nii.gz)
    DOCKER_ARGS+=(--output_space /custom_template.nii.gz)
fi

docker run --rm -t --gpus all \
    "${DOCKER_VOLUMES[@]}" \
    liuxingyu987/brainana:${version} \
    "${DOCKER_ARGS[@]}"

fs_license=/mnt/DataDrive3/xliu/prep_test/freesurfer_license.txt

# 1. prime-de
site=site-amu
# site-rockefeller
# site-mountsinaiP
# site-rockefeller
# site-uminn    

# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE/${site}
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/cpu/${site}
version=3.0.0

bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_devtest
output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_devtest_v${version}_cpu

docker run --rm -t \
    -v "$bids_dir":/input \
    -v "$output_dir":/output \
    -v "$fs_license":/fs_license.txt \
    brainana:latest \
    /input /output/preprocessed \
    -w /output/preprocessed_wd \
    --freesurfer-license /fs_license.txt

# # ------------------------------------------------------------
# # 2. with config file
# # site=site-iscmj
# # bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE/${site}
# # output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/${site}
# # config_f=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/config_iscmj.yaml

# # site=site-uncwisconsin
# # bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/UNC-Wisconsin/bids
# # output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/UNC-Wisconsin
# # config_f=${output_dir}/config.yaml

# site=site-uwmadison
# bids_dir=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE/${site}
# output_dir=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana/${site}
# config_f=${output_dir}/config.yaml

# docker run --rm -t\
#     -v "$bids_dir":/input \
#     -v "$output_dir":/output \
#     -v "$config_f":/config.yaml \
#     -v "$fs_license":/fs_license.txt \
#     brainana:latest \
#     /input /output/preprocessed \
#     -w /output/preprocessed_wd \
#     --config /config.yaml \
#     --freesurfer-license /fs_license.txt
# Methods Reference: brainana Pipeline

This document provides a detailed reference of the methods and software used at each step of the brainana preprocessing pipeline, for inclusion in manuscripts and for the report boilerplate section.

---

## Pipeline overview

Results included in this manuscript come from preprocessing performed using **brainana**, a BIDS-based, Nextflow-orchestrated pipeline for macaque (and NHP) anatomical and functional MRI. The pipeline adapts to available data and configuration: anatomical synthesis runs only when multiple T1w/T2w runs or sessions are present; slice timing correction runs only when slice timing metadata is available in BIDS; surface reconstruction is optional.

**Main software stack:**

- Python 3.11+, Nextflow; AFNI, ANTs, FSL, FreeSurfer (for optional surface reconstruction); PyTorch (for UNet skull stripping and FastSurfer-style segmentation); nibabel, pybids.
- Internal operations: **nibabel** for reading/writing NIfTI; **ANTs** for registration and bias correction; **AFNI** for reorientation, slice timing, and despiking; **FSL** for motion correction, conform (FLIRT), and mask application (fslmaths).
- Initial skullstripping (conform and functional mask): **nhp_skullstrip_nn**, fine-tuned from [NHP-BrainExtraction](https://github.com/HumanBrainED/NHP-BrainExtraction) (DeepBet).
- Anatomical segmentation: **fastsurfer_nn**, fine-tuned from [FastSurfer FastSurferCNN](https://github.com/Deep-MI/FastSurfer/tree/dev/FastSurferCNN).
- Optional surface reconstruction: **fastsurfer_surfrecon** (modified from [FastSurfer recon_surf](https://github.com/Deep-MI/FastSurfer/tree/dev/recon_surf)) and **FreeSurfer** (mri_*, mris_* tools).

---

## 1. BIDS discovery and job creation

- **Purpose:** Before the Nextflow pipeline runs, a Python discovery step scans the BIDS (and NHP-BIDS) dataset and produces structured job descriptors.
- **Methods:** Deterministic scan and grouping logic using BIDS layout and metadata; no imaging algorithms. Discovery evaluates whether anatomical **synthesis** is needed (`needs_synth`) and synthesis type/level from configuration.
- **References:** BIDS specification; pybids for layout and metadata.

---

## 2. Anatomical processing

### 2.1 Anatomical synthesis (multiple T1w/T2w)

- **When:** Multiple T1w or T2w images per session or per subject (configurable synthesis level).
- **Method:** The first file (by lexicographic order) is the fixed reference. Each other image is rigidly coregistered to the reference using **ANTs** (`antsRegistration` with rigid transform). All coregistered images (including the reference) are averaged in reference space. Output: one synthesized NIfTI per (subject, session) or per subject.
- **Tools:** ANTs (antsRegistration, antsApplyTransforms).
- **References:** Avants et al. 2008; Tustison et al. 2014 (ANTs).

### 2.2 Reorient

- **Purpose:** Standardize orientation (e.g. RAS or template orientation) before conform and registration.
- **Method:** Image reoriented to target orientation using **AFNI** `3dresample -orient`. When reorienting to a target file, orientation is read from the target and the same resample step is applied.
- **Tools:** AFNI (3dresample).
- **References:** Cox 1996 (AFNI); RRID:SCR_005927.

### 2.3 Conform to template

- **Purpose:** Align the brain to template space (same orientation and grid) for downstream registration.
- **Method:** (1) Skull-strip the input with **nhp_skullstrip_nn** (UNet) or use existing brain mask when skull stripping is disabled. (2) Pad the template; optionally downsample the template to match input resolution. (3) **FLIRT** (FSL) rigid registration (6 DOF) from brain-extracted input to template. (4) Resample template to input resolution with **AFNI** `3dresample`. (5) Apply the FLIRT transform to the full input with **FLIRT** `flirt -applyxfm` (trilinear interpolation).
- **Tools:** nhp_skullstrip_nn (PyTorch UNet), FSL FLIRT, AFNI 3dresample, fslmaths.
- **Note on nhp_skullstrip_nn:** The UNet initial skullstripping model was fine-tuned for NHP using the approach and resources from [NHP-BrainExtraction](https://github.com/HumanBrainED/NHP-BrainExtraction) (DeepBet; Wang et al. 2021).
- **References:** Jenkinson et al. 2002 (FLIRT); Smith et al. 2004 (FSL); Wang et al. 2021 (DeepBet/NHP-BrainExtraction); RRID:SCR_002823.

### 2.4 Skull stripping and segmentation (anatomical)

- **Purpose:** Brain mask and tissue/atlas segmentation for bias correction, registration, and optional surface reconstruction.
- **Method:** Segmentation in `fastsurfer_nn`: CNN-based segmentation fine-tuned on macaque MRI with CHARM and SARM level 2 atlases (ARM2 parcellation). The network architecture and training pipeline are based on [FastSurfer FastSurferCNN](https://github.com/Deep-MI/FastSurfer/tree/dev/FastSurferCNN). Produces brain mask, segmentation volume (e.g. ARM2), and optionally hemisphere mask and LUT. Brain mask is applied to the input with **FSL** `fslmaths -mul` to produce skull-stripped image.
- **Tools:** fastsurfer_nn (PyTorch; fine-tuned from FastSurferCNN), FSL fslmaths.
- **References:** Henschel et al. 2020 (FastSurfer); FSL RRID:SCR_002823.

### 2.5 Bias field correction (anatomical)

- **Purpose:** Correct intensity non-uniformity (INU) in the anatomical image.
- **Method:** **N4 bias field correction** (Tustison et al. 2010), distributed with **ANTs**. Applied with optional brain mask (`-x`); mask restricts the correction to the brain. Output: bias-corrected full-head and bias-corrected brain image.
- **Tools:** ANTs N4BiasFieldCorrection.
- **References:** Tustison et al. 2010 (N4ITK); Avants et al. 2008 (ANTs); RRID:SCR_004757.

### 2.6 Registration to template (anatomical)

- **Purpose:** Map anatomical data to a standard template space (e.g. NMT2Sym).
- **Method:** **ANTs** multi-stage registration: translation → rigid → affine → (optional) SyN. Each stage has configurable gradient step, metric (e.g. mutual information, cross-correlation, Mattes), shrink factors, convergence, and smoothing. Composite transform (and inverse) written; **antsApplyTransforms** used to resample to template space. When GPU is available and FireANTs is installed, SyN stage may use **FireANTs** (GPU); otherwise CPU **antsRegistration** is used.
- **Tools:** ANTs (antsRegistration, antsApplyTransforms); optionally FireANTs for SyN.
- **References:** Avants et al. 2008; Tustison et al. 2014 (ANTs); Klein et al. 2009 (SyN); RRID:SCR_004757.

### 2.7 T2w to T1w coregistration

- **Method:** **ANTs** rigid registration from T2w to preprocessed T1w; then **antsApplyTransforms** to resample T2w into T1w space. Interpolation from config (e.g. BSpline).
- **Tools:** ANTs (antsRegistration, antsApplyTransforms).

### 2.8 Surface reconstruction (optional)

- **Purpose:** Cortical surfaces and derived measures from brain segmentation.
- **Method:** The **fastsurfer_surfrecon** pipeline is a modified version of [FastSurfer recon_surf](https://github.com/Deep-MI/FastSurfer/tree/dev/recon_surf). It runs volume stages (bias correction with N4, mask/aseg, Talairach, normalization, WM segmentation) and surface stages (tessellation, smoothing, inflation, topology fix, parcellation, surface placement, morphometry, statistics). The pipeline **requires FreeSurfer**: it calls FreeSurfer tools (e.g. `mri_convert`, `mri_mask`, `mri_cc`, `mri_pretess`, `mri_mc`, `mri_normalize`, `mri_fill`, `mri_surf2volseg`, `mri_add_xform_to_header`, `mris_info`, `mris_smooth`, `mris_inflate`, `mris_fix_topology`, `mris_place_surface`, `mris_register`, `mris_ca_label`, `mris_anatomical_stats`) via Python wrappers. Optional T1wT2w combined image when configured. A valid FreeSurfer license is required when surface reconstruction is enabled.
- **Tools:** fastsurfer_surfrecon (modified from FastSurfer recon_surf); FreeSurfer (mri_*, mris_*); SimpleITK N4 for bias in volume stages.
- **References:** Dale et al. 1999 (surface-based analysis); Henschel et al. 2020 (FastSurfer); FreeSurfer RRID:SCR_001847.

---

## 3. Functional processing

### 3.1 Slice timing correction

- **Purpose:** Correct for slice acquisition order so that voxel time series are aligned in time.
- **Method:** **AFNI** `3dTshift`: slices are shifted in time to a reference (e.g. middle of TR). Slice timing pattern (e.g. `alt+z`, `seq+z`) is derived from BIDS `SliceTiming` and `SliceEncodingDirection`; `-tzero` and `-tpattern` are set accordingly. If slice encoding is not along z, data are swapped to z for 3dTshift then swapped back.
- **Tools:** AFNI 3dTshift.
- **References:** Cox 1996 (AFNI); RRID:SCR_005927.

### 3.2 Reorient (functional)

- **Method:** Same as anatomical reorient: **AFNI** `3dresample -orient` to target orientation (or to target file orientation).

### 3.3 Motion correction

- **Purpose:** Realign volumes to a reference to correct head motion.
- **Method:** **FSL** **mcflirt**: reference volume is either a user-specified timepoint, the middle volume, or the temporal mean (fslmaths -Tmean or fslroi). Motion correction with 6 DOF; output includes realigned 4D BOLD, transformation matrices, and motion parameters (converted to TSV). If the run has fewer than 15 volumes, motion correction is skipped and pass-through outputs (tmean, zero-filled motion params) are generated.
- **Tools:** FSL mcflirt, fslroi, fslmaths.
- **References:** Jenkinson et al. 2002 (mcflirt); FSL RRID:SCR_002823.

### 3.4 Despiking

- **Purpose:** Reduce impact of extreme timepoints (spikes) in the BOLD signal.
- **Method:** **AFNI** `3dDespike`: default or configurable -cut (c1, c2) and optional -localedit. Output: despiked 4D BOLD and optional spikiness map. Skipped if fewer than 15 volumes.
- **Tools:** AFNI 3dDespike.
- **References:** AFNI documentation; Cox 1996.

### 3.5 Within-session coregistration (optional)

- **Purpose:** Align multiple functional runs within a session to a reference run before computing session-level mean or downstream steps.
- **Method:** **ANTs** rigid or **FLIRT** rigid registration from each run’s tmean to the reference run’s tmean; then apply the transform to the full 4D BOLD and mask. Configurable method: `ants` or `flirt`.
- **Tools:** ANTs (antsRegister, antsApplyTransforms) or FSL FLIRT.

### 3.6 Bias correction (functional)

- **Method:** Same N4 approach as anatomical: **ANTs** **N4BiasFieldCorrection** applied to the temporal mean of the (motion-corrected/despiked) BOLD. Optional mask; rescale to mean 100 if configured. The bias field can be applied to the full 4D in a separate step if the pipeline supports it.
- **Tools:** ANTs N4BiasFieldCorrection.

### 3.7 Conform and skull stripping (functional)

- **Conform:** Same logic as anatomical conform: **nhp_skullstrip_nn** on the functional tmean (or skip if disabled), **FLIRT** rigid to template, **3dresample** for template grid, **flirt -applyxfm** to apply transform to 4D BOLD.
- **Skull stripping (brain mask):** **nhp_skullstrip_nn** (functional/EPI model, fine-tuned from [NHP-BrainExtraction](https://github.com/HumanBrainED/NHP-BrainExtraction)) on the temporal mean; **fslmaths -mas** to apply mask to 4D BOLD.
- **Tools:** nhp_skullstrip_nn, FSL FLIRT, AFNI 3dresample, fslmaths.

### 3.8 Registration (functional to anatomical / template)

- **Method:** **ANTs** registration (rigid, affine, or SyN as configured) from the mean functional image to the preprocessed anatomical or to the template (optionally at functional resolution). Composite transform applied to the full 4D BOLD and brain mask with **antsApplyTransforms** (e.g. BSpline interpolation for BOLD). All resamplings can be composed into a single interpolation step.
- **Tools:** ANTs (antsRegistration, antsApplyTransforms); optionally FireANTs for SyN.

---

## 4. Quality control

- **Purpose:** Visual and quantitative QC of preprocessing outputs.
- **Methods:** Snapshots generated at various steps (conform, skull strip, bias correction, anatomical-to-template and functional-to-anatomical/functional-to-template registration, motion parameters, within-session coregistration). Reports are assembled into HTML with navigation, summary, modality-specific sections, and a methods/boilerplate section.
- **Tools:** Python (nibabel, matplotlib or similar for overlays); report generation in `nhp_mri_prep.quality_control.reports`.

---

## 5. Summary table: pipeline steps and main tools

| Domain     | Step                    | Main tool / method                          |
|-----------|--------------------------|---------------------------------------------|
| Anatomical| Synthesis                | ANTs rigid + average                        |
| Anatomical| Reorient                 | AFNI 3dresample                             |
| Anatomical| Conform                  | FLIRT + nhp_skullstrip_nn + 3dresample      |
| Anatomical| Skull strip & segment    | fastsurfer_nn (FastSurferCNN fine-tuned)    |
| Anatomical| Bias correction          | ANTs N4BiasFieldCorrection                  |
| Anatomical| Registration             | ANTs (optional FireANTs for SyN)            |
| Anatomical| Surface recon (optional) | fastsurfer_surfrecon + FreeSurfer           |
| Functional| Slice timing             | AFNI 3dTshift                               |
| Functional| Reorient                 | AFNI 3dresample                             |
| Functional| Motion correction        | FSL mcflirt                                 |
| Functional| Despiking                | AFNI 3dDespike                              |
| Functional| Within-session coreg    | ANTs or FLIRT (optional)                    |
| Functional| Bias correction          | ANTs N4BiasFieldCorrection                  |
| Functional| Conform / skull strip    | FLIRT + nhp_skullstrip_nn, 3dresample      |
| Functional| Registration             | ANTs (optional FireANTs for SyN)            |

---

## 5a. Software and code sources (GitHub)

- **nhp_skullstrip_nn (initial skullstripping):** Fine-tuned from [HumanBrainED/NHP-BrainExtraction](https://github.com/HumanBrainED/NHP-BrainExtraction) (DeepBet; Wang et al. 2021).
- **fastsurfer_nn (segmentation):** Fine-tuned from [Deep-MI/FastSurfer FastSurferCNN](https://github.com/Deep-MI/FastSurfer/tree/dev/FastSurferCNN).
- **fastsurfer_surfrecon (surface reconstruction):** Modified from [Deep-MI/FastSurfer recon_surf](https://github.com/Deep-MI/FastSurfer/tree/dev/recon_surf); requires FreeSurfer.

---

## 6. References (for manuscript / boilerplate)

- Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008). Symmetric diffeomorphic image registration with cross-correlation: Evaluating automated labeling of elderly and neurodegenerative brain. *Medical Image Analysis*, 12(1), 26–41. https://doi.org/10.1016/j.media.2007.06.004
- Cox, R. W. (1996). AFNI: software for analysis and visualization of functional magnetic resonance neuroimages. *Computers and Biomedical Research*, 29(3), 162–173. https://doi.org/10.1006/cbmr.1996.0014
- Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based analysis: I. Segmentation and surface reconstruction. *NeuroImage*, 9(2), 179–194. https://doi.org/10.1006/nimg.1998.0395
- Henschel, L., Conjeti, S., Estrada, S., Diers, K., Fischl, B., & Reuter, M. (2020). FastSurfer: A fast and accurate deep learning based neuroimaging pipeline. *NeuroImage*, 219, 117012. https://doi.org/10.1016/j.neuroimage.2020.117012
- Jenkinson, M., Bannister, P., Brady, M., & Smith, S. (2002). Improved optimization for the robust and accurate linear registration and motion correction of brain images. *NeuroImage*, 17(2), 825–841. https://doi.org/10.1006/nimg.2002.1132
- Klein, A., Andersson, J., Ardekani, B. A., et al. (2009). Evaluation of 14 nonlinear deformation algorithms applied to human brain MRI registration. *NeuroImage*, 46(3), 786–802. https://doi.org/10.1016/j.neuroimage.2008.12.037
- Smith, S. M., Jenkinson, M., Woolrich, M. W., et al. (2004). Advances in functional and structural MR image analysis and implementation as FSL. *NeuroImage*, 23(S1), 208–219. https://doi.org/10.1016/j.neuroimage.2004.07.051
- Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A., Yushkevich, P. A., & Gee, J. C. (2010). N4ITK: Improved N3 bias correction. *IEEE Transactions on Medical Imaging*, 29(6), 1310–1320. https://doi.org/10.1109/TMI.2010.2046908
- Tustison, N. J., Cook, P. A., Klein, A., et al. (2014). Large-scale evaluation of ANTs and FreeSurfer cortical thickness measurements. *NeuroImage*, 99, 166–179. https://doi.org/10.1016/j.neuroimage.2014.05.044
- Wang, X., Li, X., & Xu, T. (2021). U-net model for brain extraction: Trained on humans for transfer to non-human primates. *NeuroImage*, 235, 118001. https://doi.org/10.1016/j.neuroimage.2021.118001 (NHP-BrainExtraction / DeepBet: https://github.com/HumanBrainED/NHP-BrainExtraction)

**RRIDs**

- ANTs: RRID:SCR_004757  
- AFNI: RRID:SCR_005927  
- FSL: RRID:SCR_002823  
- FreeSurfer: RRID:SCR_001847  

---

## 7. Boilerplate text for reports

The QC report “Methods” section is generated automatically from boilerplate text (fMRIPrep-style). At report generation time, the pipeline substitutes:

- **VERSION** from report metadata (the package is always referred to as brainana)
- **N_T1W**, **N_T2W**, **N_FUNC** from dataset context or from snapshot counts

**Files:**

- **`src/nhp_mri_prep/quality_control/reports.py`** — Contains `BOILERPLATE_METHODS_TEMPLATE` and `HtmlGenerator.create_methods_section()`; this is what actually fills the report’s Methods section.

`BOILERPLATE_METHODS_TEMPLATE` is the single source of truth for this text. Edit it there; for manuscript copy-paste, take the rendered Methods section out of a generated report rather than keeping a second copy in sync.

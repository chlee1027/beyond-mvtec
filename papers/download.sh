#!/usr/bin/env bash
# 논문 PDF 받기. PDF 는 저작권 때문에 레포에 올리지 않는다(.gitignore). 이 스크립트로 각자 받는다.
set -e
cd "$(dirname "$0")/pdf"
UA="Mozilla/5.0 (paper-fetch)"
get() { [ -s "$1" ] || curl -sSL -A "$UA" -o "$1" "$2"; }
get CVPR22_PatchCore_Towards_Total_Recall_in_Industrial_Anomaly_Detection.pdf https://arxiv.org/pdf/2106.08265v2
get ICPRW21_PaDiM_Patch_Distribution_Modeling_Framework_for_Anomaly_Detection_and_Localization.pdf https://arxiv.org/pdf/2011.08785v1
get CVPR19_MVTecAD_Comprehensive_Real_World_Dataset_for_Unsupervised_Anomaly_Detection.pdf https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
get CVPR20_Uninformed_Students_Student_Teacher_Anomaly_Detection_with_Discriminative_Latent_Embeddings.pdf https://arxiv.org/pdf/1911.02357v2
get ECCV22_VisA_SPot_the_Difference_Self_Supervised_Pretraining_for_Anomaly_Detection_and_Segmentation.pdf https://arxiv.org/pdf/2207.14315v1
get CVPR24_RealIAD_Real_World_Multi_View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_Detection.pdf https://arxiv.org/pdf/2403.12580v1

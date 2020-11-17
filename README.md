# Master Thesis: (Semi-)automatic fetus segmentation and visualisation in 3D ultrasound: a promising perspective
This research examines (semi-)automatic segmentation models to consistently isolate the fetus in 3D ultrasound images, and to use these segmentations to provide an objective view in visualisations techniques for an educational resource as well as a scientific purpose. The aim of the segmentation methods developed by this research is to meet or even surpass the manual segmentations received from the Amsterdam UMC and the Imperial College London. Besides that, the educational resource needs to be suitable for trainee sonographers, meaning that the tool must produce accurate and unbiased visualisations.

## Explanation of code
The workflow consists of the following phases: 
- phase 1: (Optional) preprocessing with denoising filters
- phase 2a: Heuristic segmentation models
- phase 2b: Deep learning segmentation approach: U-net
- phase 3: Volume visualisation

## System requirements 
The data structure of the datasets: 
<pre>
 datasets/ <br/>
   dataset1/ <br/>
    crop_gt/<br/>
     gt_001.dcm<br/>
     gt_002.dcm<br/>
     ...<br/>
    crop_org/<br/>
     frame_001.dcm<br/>
     frame_002.dcm<br/>
    real_org/<br/>
     original.dcm<br/>
   dataset2/<br/>
    ...<br/>
</pre> 

The code contains the following libraries which need to be installed: 
- os
- sys
- numpy 
- SimpleITK 
- pickle
- time
- tqdm
- matplotlib
- copy
- vtk
- pandas
- pydicom
- random
- jupyterthemes
- skimage
- sklearn
- warnings
- keras
- tensorflow

The computer which was used contained: 
- Microsoft Windows 10
- 16GB of system memory
- NVIDIA GeForce GTX 1050 

## Developer(s)
Romy Meester

Affiliations: University of Amsterdam (UvA)<br/>
Visualisation Lab

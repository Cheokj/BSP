# Boosting Stability and Plasticity in Class-Incremental Semantic Segmentation
This is the implementation of BSP (Boosting Stability and Plasticity in Class-Incremental Semantic Segmentation). (This repository is building....)

Our code framework is designed following the structure of Universal Model (MIA-2024) (https://github.com/ljwztc/CLIP-Driven-Universal-Model) for medical imaging experiment and CoinSeg (ICCV-2023) (https://github.com/zkzhang98/CoinSeg) for natural imaging experiment.

## Requirements
1. CUDA 11.6
2. python(3.8.0)
3. pytorch(1.13.1+cu116)
4. numpy(1.24.4)
5. einops(0.8.0)
6. monai(1.1.0)
7. matplotlib
8. pillow

## Datasets
### Multi-Organ Segmentation
LiTS：https://competitions.codalab.org/competitions/17094

MSD_task09：http://medicaldecathlon.com/

MSD_task07：http://medicaldecathlon.com/

KiTS：https://kits19.grand-challenge.org/

AMOS：https://amos22.grand-challenge.org/

BTCV：https://www.synapse.org/Synapse:syn3193805/wiki/89480

### Natural Imaging Segmentation
Download VOC 2012 by running ./natural_imaging_segmentation/datasets/data/download_voc.sh

Organize datasets in the following structure.
<pre> path_to_your_dataset/ 
  ├── VOC2012/
  ├── Annotations/ 
  │ ├── ImageSet/ 
  │ ├── JPEGImages/ 
  │ ├── SegmentationClassAug/ 
  │ └── proposal100/  
</pre>
You can get proposal100 here (provided by MicroSeg) (https://github.com/zkzhang98/MicroSeg)
## Run

## Experimental results

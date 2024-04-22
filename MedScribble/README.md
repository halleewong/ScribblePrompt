
# MedScribble

> **tl;dr:** `MedScribble` is a collection of diverse biomedical image segmentation tasks with multi-annotator scribble annotations

We collected manual scribbles from 3 annotators for 14 segmentation tasks from 14 different open-access biomedical image segmentation datasets. MedScribble contains a total of 64 2D image-segmentation pairs with 3 sets of scribble annotations for each image-segmentation pair. 

For each segmentation task (i.e., dataset/label combination) the annotators were shown five training examples with the ground truth segmentation per task and instructed to draw positive and negative scribbles to indicate the region of interest on new images. 

Annotators drew the scribbles in a Gradio web app. Annotators 1 and 2 used an iPad with stylus and Annotator 3 used a laptop trackpad to draw scribbles. 

All images were padded square (with zeros) before being resized to 256x256 and rescaled to [0,1]. For 3D datasets, we took either the middle slice (`midslice`) or slice with maximum label area (`maxslice`) as indicated by the folder name.

Each example folder is structured as follows:
```
img.npy
seg.npy
scribble_1.npy
scribble_2.npy
scribble_3.npy
```
The scribble annotations (`scribble_X.npy`) for each annotator are stored as a 2x256x256 array where the 1st channel contains positive scribble (on the label) and the 2nd channel contains negative scribbles (on the background).

See `tutorial.ipynb` for a simple dataloader and preview of the data.

# Data Sources

See the list of data sources below for the sources of the images and segmentations. 

For `AbdominalUS` and `SpineWeb`, you will need to follow the instructions on the respective websites to retrieve data and then perform the following processing steps:

1. Resample the volume (and segmentation) to 1mmx1mmx1mm spacing 
2. Pad the volume (and segmentation) with zeros to be square
2. Rescale the values to be on [0,1]
3. Take the middle slice along axis 0

For the other datasets, which permit redistribution, we provide the relevant slices and segmentations. 

# References

If you use this dataset, please cite:
```
@article{wong2024scribbleprompt,
  title={ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image},
  author={Hallee E. Wong and Marianne Rakic and John Guttag and Adrian V. Dalca},
  journal={arXiv:2312.07381},
  year={2024},
}
```

For the **images and segmentations**, please refer to and cite the original data sources (below). 

## AbdominalUS

website: https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm

citation:
```
@article{AbdominalUS,
  title={Improving realism in patient-specific abdominal ultrasound simulation using CycleGANs},
  author={Vitale, Santiago and Orlando, Jos{\'e} Ignacio and Iarussi, Emmanuel and Larrabide, Ignacio},
  journal={International journal of computer assisted radiology and surgery},
  volume={15},
  number={2},
  pages={183--192},
  year={2020},
  publisher={Springer}
}
```

## ACDC

website: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

citation:
```
@article{ACDC,
  title={Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Clement and Cervenansky, Frederick and Yang, Xin and Heng, Pheng-Ann and Cetin, Irem and Lekadir, Karim and Camara, Oscar and Ballester, Miguel Angel Gonzalez and others},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={11},
  pages={2514--2525},
  year={2018},
  publisher={ieee}
  doi={10.1109/TMI.2018.2837502}
}
```

## BTCV 

website: https://www.synapse.org/#!Synapse:syn3193805/wiki/217790

license: [CC by 4.0](https://creativecommons.org/licenses/by/4.0/)

citation:
```
@inproceedings{BTCV,
  title={Miccai multi-atlas labeling beyond the cranial vault--workshop and challenge},
  author={Landman, Bennett and Xu, Zhoubing and Igelsias, J and Styner, Martin and Langerak, T and Klein, Arno},
  booktitle={Proc. MICCAI Multi-Atlas Labeling Beyond Cranial Vault Workshop Challenge},
  volume={5},
  pages={12},
  year={2015}
}
```

## CAMUS

website: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html

citation:
```
@article{CAMUS,
  title={Deep learning for segmentation using an open large-scale dataset in 2D echocardiography},
  author={Leclerc, Sarah and Smistad, Erik and Pedrosa, Joao and {\O}stvik, Andreas and Cervenansky, Frederic and Espinosa, Florian and Espeland, Torvald and Berg, Erik Andreas Rye and Jodoin, Pierre-Marc and Grenier, Thomas and others},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={9},
  pages={2198--2210},
  year={2019},
  publisher={IEEE}
}
```

## CHAOS

webiste: https://zenodo.org/records/3431873

license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

citation:
```
@misc{CHAOS,
  author       = {Ali Emre Kavur and M. Alper Selver and Oğuz Dicle and Mustafa Barış and  N. Sinem Gezer},
  title        = {{CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge Data}},
  month        = Apr,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1.03},
  doi          = {10.5281/zenodo.3362844},
  url          = {https://doi.org/10.5281/zenodo.3362844}
}
```

## HipXRay

website: https://data.mendeley.com/datasets/zm6bxzhmfz/1

license: [CC by 4.0](https://creativecommons.org/licenses/by/4.0/)

```
@article{HipXRay,
	title = {X-ray images of the hip joints},
	volume = {1},
	url = {https://data.mendeley.com/datasets/zm6bxzhmfz/1},
	doi = {10.17632/zm6bxzhmfz.1},
	language = {en},
	urldate = {2023-09-03},
	author = {Gut, Daniel},
	month = jul,
	year = {2021},
	note = {Publisher: Mendeley Data},
}
```

## OASIS

website: https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md

citation:

```
@article{OASIS-data,
  title={Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults},
  author={Marcus, Daniel S and Wang, Tracy H and Parker, Jamie and Csernansky, John G and Morris, John C and Buckner, Randy L},
  journal={Journal of cognitive neuroscience},
  volume={19},
  number={9},
  pages={1498--1507},
  year={2007},
  publisher={MIT Press}
}

@article{OASIS-proccessing,
    title = "Learning the Effect of Registration Hyperparameters with HyperMorph",
    author = "Hoopes, Andrew and Hoffmann, Malte and Greve, Douglas N. and Fischl, Bruce and Guttag, John and Dalca, Adrian V.",
    journal = "Machine Learning for Biomedical Imaging",
    volume = "1",
    issue = "IPMI 2021 special issue",
    year = "2022",
    pages = "1--30",
    issn = "2766-905X",
    url = "https://melba-journal.org/2022:003"
}
```

## OCTA500

website: https://ieee-dataport.org/open-access/octa-500

license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

citation:
```
@article{OCTA500,
  title={Ipn-v2 and octa-500: Methodology and dataset for retinal image segmentation},
  author={Li, Mingchao and Zhang, Yuhan and Ji, Zexuan and Xie, Keren and Yuan, Songtao and Liu, Qinghuai and Chen, Qiang},
  journal={arXiv preprint arXiv:2012.07261},
  year={2020}
}
```

## PanDental

website: https://data.mendeley.com/datasets/hxt48yk462/2

license: [CC BY NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)

citation:
```
@article{PanDental,
  title={Automatic segmentation of mandible in panoramic x-ray},
  author={Abdi, Amir Hossein and Kasaei, Shohreh and Mehdizadeh, Mojdeh},
  journal={Journal of Medical Imaging},
  volume={2},
  number={4},
  pages={044003},
  year={2015},
  publisher={SPIE}
}
```

## PAXRay

website: https://constantinseibold.github.io/paxray/

license: [Attribution-NonCommercial-ShareAlike 4.0 International](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/?tab=License-1-ov-file)

citation:
```
@inproceedings{PAXRay,
    author    = {Seibold,Constantin and Reiß,Simon and Sarfraz,Saquib and Fink,Matthias A. and Mayer,Victoria and Sellner,Jan and Kim,Moon Sung and Maier-Hein, Klaus H.  and Kleesiek, Jens  and Stiefelhagen,Rainer}, 
    title     = {Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding}, 
    booktitle = {Proceedings of the 33th British Machine Vision Conference (BMVC)},
    year  = {2022}
}
```

## SCD

website: https://www.cardiacatlas.org/sunnybrook-cardiac-data/

license: [CC0 1.0 DEED](https://creativecommons.org/publicdomain/zero/1.0/)

citation:
```
@article{SCD,
  title={Evaluation framework for algorithms segmenting short axis cardiac MRI},
  author={Radau, Perry and Lu, Yingli and Connelly, Kim and Paul, Gideon and Dick, AJWG and Wright, Graham},
  journal={The MIDAS Journal-Cardiac MR Left Ventricle Segmentation Challenge},
  volume={49},
  year={2009}
}

```

## SpineWeb (Dataset 7)

website: http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_7.3A_Intervertebral_Disc_Localization_and_Segmentation.3A_3D_T2-weighted_Turbo_Spin_Echo_MR_image_Database

citation:
```
@article{SpineWeb,
  title={Evaluation and comparison of 3D intervertebral disc localization and segmentation methods for 3D T2 MR data: A grand challenge},
  author={Zheng, Guoyan and Chu, Chengwen and Belav{\`y}, Daniel L and Ibragimov, Bulat and Korez, Robert and Vrtovec, Toma{\v{z}} and Hutt, Hugo and Everson, Richard and Meakin, Judith and Andrade, Isabel L{\u{o}}pez and others},
  journal={Medical image analysis},
  volume={35},
  pages={327--344},
  year={2017},
  publisher={Elsevier}
}
```

## STARE

webiste: https://cecas.clemson.edu/~ahoover/stare/

citation:
```
@article{STARE,
  title={Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response},
  author={Hoover, AD and Kouznetsova, Valentina and Goldbaum, Michael},
  journal={IEEE Transactions on Medical imaging},
  volume={19},
  number={3},
  pages={203--210},
  year={2000},
  publisher={IEEE}
}
```

## WBC

website: https://github.com/zxaoyou/segmentation_WBC

license: [GNU General Public License v3.0](https://github.com/zxaoyou/segmentation_WBC/blob/master/LICENSE)

citation:
```
@article{WBC,
  title={Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning},
  author={Xin Zheng and Yong Wang and Guoyou Wang and Jianguo Liu},
  journal={Micron},
  volume={107},
  pages={55--71},
  year={2018},
  publisher={Elsevier}
  doi={https://doi.org/10.1016/j.micron.2018.01.010},
  url={https://www.sciencedirect.com/science/article/pii/S0968432817303037}
}
```
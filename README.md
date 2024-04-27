# InterCap
Source code for the GCPR 2022 paper: &lt;InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction>

# License
Software Copyright License for non-commercial scientific research purposes. Please read carefully the terms and conditions and any accompanying documentation before you download and/or use the InterCap model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License

# Dependencies 
This code is based on [***PROX***](https://github.com/mohamedhassanmus/prox). Please check the ***Dependencies*** section for the necesssary dependencies.
Note that both ***SMPL*** and ***SMPLX*** models need to be downloaded and placed into folder ***models***.

# InterCap dataset
To run the code, one needs to get the InterCap dataset from the [***InterCap website***](https://intercap.is.tue.mpg.de/index.html). Please download the ***RGBD_Images*** zip files, uncompress it, then do data pre-processing (keypoint detection and segmentation) and save the results into the following folders:
- Keypoints: one subfolder for each motion sequence. Inside the subfolder, keypoint json files are sotred in spearation for each camera view. Sample folder structure shown in ***Data/keypoints/subject01_motion01_seg01***. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is used for 2D keypoint detection.
- BodyIndex: similarly, one subfolder for each motion sequence, the segmentation images for each camera view are sotred in a individual folder. Sample folder structure shown in ***Data/recordings/subject01_motion01_seg01/BodyIndexColor_X?***. [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) is adopted for this purpose. The segmentation results for the body need to be saved into png image format. Check ***/is/cluster/work/yhuang2/Projects/PROX_D_Clean/Data/recordings/subject01_motion01_seg01*** for the folder structure.
- mrcnn_res: the PointRend result for each image is needed to run object tracking. Please save the results inside ***../Data/mrcnn***.
- Depth: depth images need also be organized in a similar way. Sample folder structure shown in: ***Data/recordings/subject01_motion01_seg01/Depth_X***.

# How to run the code    
- Step 1: run object fitting on the pre-processed data
  Go to the subfolder ***obj_track***, then run ***run.sh*** inside the folder. The ***obj_track/configs*** contain initialization values of pose and translation for each sequence, with the name of each figure indicating the corresponding sequence. After running, there will be a res.pkl file inside the sequnce folder.

- Step 2: run per-frame body fitting on the pre-processed data
  Go to the subfolder ***prox_first***, then run ***run.sh***. This script will save both a pkl file and ply/obj file for each frame.

- Step 3: run joint fitting to ensure body-object contact
  To support a bigger batch size, we reduce the number of vertices for SMPLX model. To do this, please find where SMPLX model resides, then add these lines of code at the beginning of 
  ```
  with open('../models/wo_head.pkl', 'rb') as fin:
      import pickle as pkl
      wo_head = pkl.load(fin)
      wo_head = wo_head['wo_head']
  ```
  Then add these lines of code _vertices = vertices[:, wo_head, :]_ right before the line of _output = SMPLXOutput(_.
  ***run.sh** inside folder ***prox_second*** can be run now. It gives the results for the whole sequences, enforcing contact between human and object.

# License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YinghaoHuang91/InterCap/blob/master/License) file. 
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/YinghaoHuang91/InterCap/blob/master/License).



# Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{huang2022intercap,
    title={{InterCap}: Joint Markerless 3D Tracking of Humans and Objects in Interaction},
    author={Huang, Yinghao and Tehari, Omid and Black, Michael J. and Tzionas, Dimitrios},
    booktitle={German Conference on Pattern Recognition},
    year={2022},
    organization={Springer}
}

@article{huang2024intercap,
    title={InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction from Multi-view RGB-D Images},
    author={Huang, Yinghao and Taheri, Omid and Black, Michael J and Tzionas, Dimitrios},
    journal={International Journal of Computer Vision},
    pages={1--16},
    year={2024},
    publisher={Springer}
}
```


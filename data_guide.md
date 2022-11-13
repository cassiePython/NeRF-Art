# Data Convention
## From Images
If you have a set of multiple view images, you need to run [COLMAP](https://colmap.github.io/) to estimate the camera parameters. You can do it in COLMAP GUI or try `data/img2poses.py`. You may also refer to [nerf_pl](https://github.com/kwea123/nerf_pl#your-own-data) for more explanation on this step.

Additionally, if you want to reconstruction with NeuS, you might need to extract the foreground matte from the images. This could be achieve by various methods. We provide a colab notebook [here].

## From Colmap
If you have the camera parameters estimated by COLMAP with structure like following, you need to convert the COLMAP format to VolSDF format and NeuS format respectively.
```
scene/
├── sparse_points.ply
├── images/
├── sparse/
    ├── 0/
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin

```

### to VolSDF format
Try following command:
```
python colmap2volsdf.py --root_dir $YOUR_SCENE_DIR 
```

For a more detailed usage, please refer to VolSDF's [data convention](https://github.com/lioryariv/volsdf/blob/main/DATA_CONVENTION.md).

### to NeuS format
Please refer to NeuS's [data convention](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data#option-2-use-colmap).


<!-- ### Other Useful Information
Our code is build upon [neurecon](https://github.com/ventusff/neurecon) and therefore you may also refer to their guide to process your own data. -->
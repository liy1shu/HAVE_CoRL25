This folder includes code you need to generate special assets for ambiguity tests: multimodal ambiguous door, and uneven objects with different mass centers

## Generate Multimodal Doors

```
chmod +x scripts/generate_assets/generate_multimodal_door.sh

scripts/generate_assets/generate_multimodal_door.sh 
/home/yishu/datasets/partnet-mobility /home/yishu/datasets/failure_history_door_release_test
```

set the save_video flag to True to also save the videos for each created instance for visualization

## Generate Uneven Assets

Prestep: Find your python package pybullet_data path, and put panda_with_spatula.urdf under your pybullet_data/franka_panda/，and Spatula.obj in pybullet_data/franka_panda/meshes

You need to create a dataset folder (e.g. uneven/), create a folder named raw/ under it, put bookmark1/2.obj, knife.obj under raw/.

Then run the generate_uneven.sh to automatically generate assets with different mass distribution:

```
chmod +x generate_uneven.sh
```

execute under root project path

```
scripts/generate_assets/generate_uneven.sh ~/datasets/unevenobject
```

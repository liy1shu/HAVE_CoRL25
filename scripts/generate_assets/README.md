This folder includes code you need to generate special assets for ambiguity tests: multimodal ambiguous door, and uneven objects with different mass centers

## Generate Multimodal Doors

chmod +x scripts/generate_assets/generate_multimodal_door.sh

scripts/generate_assets/generate_multimodal_door.sh /home/yishu/datasets/partnet-mobility /home/yishu/datasets/failure_history_door_release_test

set the save_video flag to True to also save the videos for each created instance for visualization

## Generate Uneven Assets

You need to create a dataset folder (e.g. uneven/), create a folder named raw/ under it, put bookmark1/2.obj, knife.obj under raw/.

Then run the generate_uneven.sh to automatically generate assets with different mass distribution:

chmod +x generate_uneven.sh

execute under root project path

```
scripts/generate_assets/generate_uneven.sh ~/datasets/unevenobject
```

TODO
关于urdf还有一个是要把panda_with_spatula.urdf放到pybullet_data/franka_panda/目录下，并且把Spatula.obj放到pybullet_data/franka_panda/meshes
# Change ~/datasets/unevenobject/ to your desired dataset path (should have raw/ folder with obj files inside)
DATA_PATH=$1

echo "Using dataset path: $DATA_PATH"

# Run the generation commands using the DATA_PATH variable
python scripts/generate_assets/generate_uneven.py --save_path "$DATA_PATH/toy" --obj_type toy
python scripts/generate_assets/generate_uneven.py --save_path "$DATA_PATH/test" --obj_type bookmark1 --num_urdf 20 --template_urdf "assets/bookmark1_template.urdf"
python scripts/generate_assets/generate_uneven.py --save_path "$DATA_PATH/test" --obj_type bookmark2 --num_urdf 20 --template_urdf "assets/bookmark2_template.urdf"
python scripts/generate_assets/generate_uneven.py --save_path "$DATA_PATH/test" --obj_type knife --num_urdf 20 --template_urdf "assets/knife_template.urdf"

PM_PATH=$1
MD_PATH=$2

echo "Reading partnet-mobility dataset from path: $PM_PATH"
echo "Output multimodal door dataset to path: $MD_PATH"

python scripts/generate_assets/generate_multimodal_door.py --pm_path "$PM_PATH" --output_md_path "$MD_PATH" --save_video False
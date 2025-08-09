import argparse
import os
import random
import pybullet_data

def generate_urdf(save_path, num_urdf, template_urdf):
    os.makedirs(save_path, exist_ok=True)
    
    length_range = (0.4, 1.0)
    width_range = (0.02, 0.05)
    height_range = (0.02, 0.05)

    for i in range(num_urdf):
        if i % 10 == 0:
            length_range = (0.4, 0.6)
        elif i % 10 <= 6:
            length_range = (0.6, 0.8)
        else:
            length_range = (0.8, 1.0)
            
        length = round(random.uniform(length_range[0], length_range[1]),2)
        width = round(random.uniform(width_range[0], width_range[1]),2)
        height = round(random.uniform(height_range[0], width),2)

        alpha = 0.9
        x_range = round(alpha*length/2-0.05, 2) # a magic number. cuboid is 0.01*2 in y-axis and spatula 0.0346*2

        com_x = round(random.uniform(-x_range, x_range),2)
        com_y = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)
        com_z = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)

        urdf_dir = os.path.join(save_path, str(i))
        os.makedirs(urdf_dir, exist_ok=True)
        urdf_path = os.path.join(urdf_dir, 'rod.urdf')

        with open(f'{template_urdf}', 'r') as f:
            template = f.read()

        template = template.replace('template_name', f'rod_{i}', 1)
        template = template.replace('template_scale', f'{length} {width} {height}', 2)
        template = template.replace('template_xyz', f'{com_x} {com_y} {com_z}', 1)

        with open(urdf_path, 'w') as f:
            f.write(template)

        print(f'Generated URDF file: {urdf_path}')

def generate_urdf_obj(save_path, num_urdf, template_urdf, obj_type):
    os.makedirs(save_path, exist_ok=True)
    
    import pybullet as pb
    physicsClientId = pb.connect(pb.DIRECT)
    tpid = pb.loadURDF(template_urdf)
    xyz_offset = pb.getAABB(tpid, physicsClientId=physicsClientId)
    x_offset = (xyz_offset[1][0] - xyz_offset[0][0])
    y_offset = (xyz_offset[1][1] - xyz_offset[0][1])
    z_offset = (xyz_offset[1][2] - xyz_offset[0][2])
    
    if obj_type == "bookmark2":
        length_range_org = (0.5, 0.9)
        width_range = (0.05, 0.09)
        height_range = (0.01, 0.03)
    elif obj_type == "bookmark1":
        length_range_org = (0.5, 1.0)
        width_range = (0.08, 0.14)
        height_range = (0.02, 0.04)
    elif obj_type == "knife":
        length_range_org = (0.5, 1.0)
        width_range = (0.08, 0.14)
        height_range = (0.01, 0.015)

    for i in range(num_urdf):
        if i % 10 == 0:
            length_range = (max(length_range_org[0], 0.4*length_range_org[1]), 0.6*length_range_org[1])
        elif i % 10 <= 6:
            length_range = (0.6*length_range_org[1], 0.8*length_range_org[1])
        else:
            length_range = (0.8*length_range_org[1], 1.0*length_range_org[1])
        
        length = round(random.uniform(length_range[0], length_range[1]),2)
        width = round(random.uniform(width_range[0], width_range[1]),2)
        height = round(random.uniform(height_range[0], min(height_range[1], width)),2)

        alpha = 0.9
        x_range = round(alpha*length/2-0.05, 2) # a magic number. cuboid is 0.01*2 in y-axis and spatula 0.0346*2

        com_x = round(random.uniform(-x_range, x_range),2)
        com_y = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)
        com_z = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)

        urdf_dir = os.path.join(save_path, f'{obj_type}_{i}')
        os.makedirs(urdf_dir, exist_ok=True)
        urdf_path = os.path.join(urdf_dir, f'rod.urdf')

        with open(f'{template_urdf}', 'r') as f:
            template = f.read()

        template = template.replace(f'{obj_type}', f'{obj_type}_{i}', 1)
        template = template.replace(f'filename="{obj_type}.obj"', f'filename="../../{obj_type}.obj"', 2)
        template = template.replace('scale="1.0 1.0 1.0"', f'scale="{length/x_offset:.4f} {width/y_offset:.4f} {height/z_offset:.4f}"', 2)
        template = template.replace('xyz="0 0 0"', f'xyz="{com_x:.2f} {com_y:.2f} {com_z:.2f}"', 1)
        with open(urdf_path, 'w') as f:
            f.write(template)

        print(f'Generated URDF file: {urdf_path}')
    pb.disconnect()
        
def generate_urdf_toy(save_path, template_urdf):
    os.makedirs(save_path, exist_ok=True)
    
    length = 1.0
    width = 0.04
    height = 0.02

    for alpha_idx in range(2, 19):
        alpha = (alpha_idx-10)/10

        com_x = alpha*length/2
        com_y = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)
        com_z = 0 # round(random.uniform(center_of_mass_range[0], center_of_mass_range[1]),2)

        urdf_dir = os.path.join(save_path, str(alpha_idx))
        os.makedirs(urdf_dir, exist_ok=True)
        urdf_path = os.path.join(urdf_dir, 'rod.urdf')

        with open(f'{template_urdf}', 'r') as f:
            template = f.read()

        template = template.replace('template_name', f'rod_{alpha_idx}', 1)
        template = template.replace('template_scale', f'{length} {width} {height}', 2)
        template = template.replace('template_xyz', f'{com_x} {com_y} {com_z}', 1)

        with open(urdf_path, 'w') as f:
            f.write(template)

        print(f'Generated URDF file: {urdf_path}')
        
def arg_parser():
    parser = argparse.ArgumentParser(description='Generate URDF files with random parameters.')
    parser.add_argument('--save_path', type=str, default= "train", help='The path to save urdf.')
    parser.add_argument('--num_urdf', type=int, default= 1, help='The number of URDF files to generate.')
    parser.add_argument('--obj_type', type=str, default= "rod", help='rod, knife, bookmark1, bookmark2.')
    parser.add_argument('--template_urdf', type=str, default= "assets/template.urdf", help='The path to your template urdf.')
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    save_dir = 'path to save dir'
    args.save_path = os.path.join(save_dir, args.save_path)
    if args.obj_type == "rod":
        generate_urdf(args.save_path, args.num_urdf, args.template_urdf)
    elif args.obj_type == "toy":
        generate_urdf_toy(args.save_path, args.template_urdf)
    else:
        generate_urdf_obj(args.save_path, args.num_urdf, args.template_urdf, args.obj_type)
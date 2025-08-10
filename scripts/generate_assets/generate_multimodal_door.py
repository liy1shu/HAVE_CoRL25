import xml.etree.ElementTree as ET
import shutil
import os
import math
import json
from tqdm import tqdm
from have.env.articulated.simulation import *
import trimesh
import torch
import numpy as np
import pybullet as p
import imageio
import argparse

def adjust_door_limits(input_urdf_path, output_urdf_path, link_name):
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()
    
    # Find the joint associated with the link
    joint_name = link_name.replace('link', 'joint')
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Joint {joint_name} not found")
    
    # Get joint parameters
    joint_origin = joint.find('origin')
    axis = joint.find('axis')
    limit = joint.find('limit')
    original_lower = float(limit.get('lower'))
    original_upper = float(limit.get('upper'))
    joint_axis = [float(x) for x in axis.get('xyz').split()]
    
    # Find the link and its visual/collision elements
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link {link_name} not found")

    # Calculate rotation parameters (assuming Y-axis rotation)
    rotation_angle = -original_lower
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    # Process all visual and collision elements
    for elem_type in ['visual', 'collision']:
        for elem in link.findall(elem_type):
            origin = elem.find('origin')
            if origin is None:
                origin = ET.Element('origin', xyz="0 0 0", rpy="0 0 0")
                elem.insert(0, origin)
            
            # Get current values
            xyz = [float(x) for x in origin.get('xyz').split()]
            rpy = [float(x) for x in origin.get('rpy').split()] if 'rpy' in origin.attrib else [0.0, 0.0, 0.0]
            
            # Rotate position (around Y-axis)
            x, y, z = xyz
            new_x = x * cos_theta - z * sin_theta
            new_z = x * sin_theta + z * cos_theta
            new_xyz = [new_x, y, new_z]
            neg_new_xyz = [-1 * new_x, -1 * y, -1 * new_z]
            print(x, y, z)
            print(new_x, y, new_z)
            
            # Adjust orientation (add rotation to pitch)
            new_rpy = [rpy[0], rpy[1] - rotation_angle, rpy[2]]
            
            # Update element
            origin.set('xyz', ' '.join(map(str, new_xyz)))
            origin.set('rpy', ' '.join(map(str, new_rpy)))
            # joint_origin.set('xyz', ' '.join(map(str, neg_new_xyz)))

    # Update joint limits
    new_lower = 0.0
    new_upper = original_upper - original_lower
    limit.set('lower', str(new_lower))
    limit.set('upper', str(new_upper))

    # Save modified URDF
    tree.write(output_urdf_path)
    print(f"Saved adjusted URDF to {output_urdf_path}")


def get_merged_vertices(link, input_urdf_dir):
    """
    Extracts and merges all vertices from multiple visual geometries within a link.
    
    Parameters:
    - link: XML element representing a link in the URDF.
    - input_urdf_dir: Directory where the mesh files are located.
    
    Returns:
    - A NumPy array containing all merged vertices.
    """
    all_vertices = []

    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh_element = geometry.find("mesh")
            if mesh_element is not None:
                mesh_path = mesh_element.get("filename")
                full_path = os.path.join(input_urdf_dir, mesh_path)
                
                if os.path.exists(full_path):
                    # Load mesh and store vertices
                    mesh = trimesh.load(full_path, force='mesh')
                    all_vertices.append(mesh.vertices)
                else:
                    print(f"Warning: Mesh file {full_path} not found.")

    if all_vertices:
        # Concatenate all vertices into a single array
        return np.vstack(all_vertices)
    else:
        return np.array([])  # Return an empty array if no meshes were found

    

def modify_urdf(input_urdf_dir, output_root_dir, output_prefix, link_id, handle=True, axis_change=False, skip_first=False):
    input_urdf = os.path.join(input_urdf_dir, 'mobility.urdf')
    tree = ET.parse(input_urdf)
    root = tree.getroot()
    joint_id = link_id.replace('link', 'joint')

    # Find joint_1 (the door hinge)
    joint = root.find(f".//joint[@name='{joint_id}']")
    if joint is None:
        raise ValueError(f"Joint '{joint_id}' not found in URDF")

    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find('limit')
    original_lower = float(limit.get('lower'))
    original_upper = float(limit.get('upper'))

    # Find link_1 (the door panel) for visual and collision adjustments
    link = root.find(f".//link[@name='{link_id}']")
    # geometry = link.find("visual").find("geometry")
    # if geometry is not None:
    #     mesh_path = geometry.find("mesh").get("filename")
    #     # Load mesh
    #     mesh = trimesh.load(os.path.join(input_urdf_dir, mesh_path), force='mesh')
    #     vertices = mesh.vertices
    vertices = get_merged_vertices(link, input_urdf_dir)
    # print("Merged vertices shape:", vertices.shape)


    if link is None:
        raise ValueError(f"Link '{link_id}' not found in URDF")
    # print(link.find("visual").find("origin").get("rpy"))
    # rpy = float(link.find("visual").find("origin").get("rpy").split()[1])

    # Extract original joint parameters
    original_xyz = [float(x) for x in origin.get("xyz").split()]
    original_axis = [float(x) for x in axis.get("xyz").split()]

    # Extract original visual and collision origins
    visual_origins = [vis.find("origin") for vis in link.findall("visual") if vis.find("origin") is not None]
    collision_origins = [col.find("origin") for col in link.findall("collision") if col.find("origin") is not None]

    visual_xyz_list = [[float(x) for x in v.get("xyz").split()] for v in visual_origins]
    collision_xyz_list = [[float(x) for x in c.get("xyz").split()] for c in collision_origins]

    def judge_default_axis_side(link_xyz):  # This should be the link_door origin
        
        angle_rad = -original_axis[1] * original_lower
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, -np.sin(angle_rad)],
            [0, 1, 0],
            [np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])  # counterclockwise
        # Apply rotation
        rotated_vertices = (vertices + np.array(link_xyz)[np.newaxis, :]) @ rotation_matrix.T
        min_x_id = np.argmin(rotated_vertices[:, 0])
        max_x_id = np.argmax(rotated_vertices[:, 0])
        door_width = np.abs(rotated_vertices[min_x_id, 0] - rotated_vertices[max_x_id, 0])

        min_x = np.min(rotated_vertices[:, 0])
        max_x = np.max(rotated_vertices[:, 0])

        # print(link_xyz, min_x, max_x, np.min(vertices[:, 0]), np.max(vertices[:, 0]))
        # print(original_lower, original_axis[1])
        default_is_right = abs(0 - max_x) < abs(0 - min_x)  # Originally it was max:

        return default_is_right, door_width

    default_is_right, door_width = judge_default_axis_side(visual_xyz_list[0])
    print(default_is_right, door_width)

    def transform_door_link_origin(link_xyz):
        x, y, z = link_xyz
        new_xyz = [None, y, None]
        x_diff = np.abs(door_width * np.cos(original_lower))
        # print("X_diff:", x_diff)
        new_xyz[0] = (x + x_diff) if default_is_right else (x - x_diff)

        # if default_is_right:  # Originally was max (right), now it should be left
        #      #, y, -vertices[min_x_id, 2]]
        # else:
        #     new_xyz = [x - x_diff, y, -vertices[max_x_id, 2]]
        z_diff = np.abs(door_width * np.sin(original_lower))
        orig_z = -z
        min_z = np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])
        default_is_back = abs(orig_z - min_z) < abs(orig_z - max_z)  # Originally it was max:
        # print("Z_diff:", z_diff)
        new_xyz[2] = (z - z_diff) if default_is_back else (z + z_diff)
        # x, y, z = transform_to_absolute(new_xyz)
        return new_xyz
    
    def transform_joint_origin(joint_xyz):
        # Judge whether the default mode is on right / left
        x, y, z = joint_xyz

        if default_is_right:
            return [x - door_width, y, z]
        else:
            return [x + door_width, y, z]

    # Define the four modes
    modes = {
        "1": {  # Push Right (original)
            "joint_xyz": original_xyz,
            "axis": original_axis,
            "visual_xyz_list": visual_xyz_list,
            "collision_xyz_list": collision_xyz_list,
            "lower": original_lower,
            "upper": original_upper,
        },
        "2": {  # Pull Right
            "joint_xyz": original_xyz,
            "axis": [original_axis[0], -original_axis[1], original_axis[2]],
            "visual_xyz_list": visual_xyz_list,
            "collision_xyz_list": collision_xyz_list,
            "lower": -original_lower,
            "upper": original_upper - 2 * original_lower,
        },
        "3": {  # Pull Left
            "joint_xyz": transform_joint_origin(original_xyz),
            "axis": original_axis,
            "visual_xyz_list": [transform_door_link_origin(v) for v in visual_xyz_list],
            "collision_xyz_list": [transform_door_link_origin(c) for c in collision_xyz_list],
            "lower": original_lower,
            "upper": original_upper,
        },
        "4": {  # Push Left
            "joint_xyz": transform_joint_origin(original_xyz),
            "axis": [original_axis[0], -original_axis[1], original_axis[2]],
            "visual_xyz_list": [transform_door_link_origin(v) for v in visual_xyz_list],
            "collision_xyz_list": [transform_door_link_origin(c) for c in collision_xyz_list],
            "lower": -original_lower,
            "upper": original_upper - 2 * original_lower,
        },
    }

    # Fix all of the joints except for this one: (For double door)
    for other_joint in root.findall("joint"):
        if other_joint.get("name") != joint_id:
            if other_joint.get("type") == "revolute":
                other_limit = other_joint.find("limit")
                other_lower_limit = other_limit.get("lower")
                other_limit.set('lower', '0')
                other_lower_limit = float(other_lower_limit)  # Convert to float

                other_origin = other_joint.find("origin")
                rot_direction = float(other_joint.find("axis").get("xyz").split()[1])
                other_origin.set("rpy", f"0 {rot_direction * other_lower_limit} 0")  # Assume rotation around x-axis
            other_joint.set("type", "fixed")
        else:
            other_joint.set("type", "revolute")

    for mode, values in list(modes.items()):
        if skip_first:
            if int(mode) == 1:
                continue
        if handle or not axis_change:  # Only do push & pull
            if int(mode) >= 3:
                continue
        # Modify joint parameters
        origin.set("xyz", " ".join(map(str, values["joint_xyz"])))
        axis.set("xyz", " ".join(map(str, values["axis"])))
        limit.set('lower', str(values['lower']))
        limit.set('upper', str(values['upper']))

        # Modify visual origins
        for v_origin, new_xyz in zip(visual_origins, values["visual_xyz_list"]):
            v_origin.set("xyz", " ".join(map(str, new_xyz)))

        # Modify collision origins
        for c_origin, new_xyz in zip(collision_origins, values["collision_xyz_list"]):
            c_origin.set("xyz", " ".join(map(str, new_xyz)))

        # Save the modified URDF
        output_dir = os.path.join(output_root_dir, f"{output_prefix}{'_handle' if handle else ''}_{mode}")
        output_file = os.path.join(output_dir, "mobility.urdf")

        print(input_urdf_dir, output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(input_urdf_dir, output_dir, dirs_exist_ok=True)  # Ensure directory exists
        os.remove(output_file)  # Remove old URDF file if it exists
        tree.write(output_file)

        print(f"Saved: {output_file}")


def create_no_handle_replicate(input_urdf_dir, link_list, output_dir):
    # Save the modified URDF

    urdf_path = os.path.join(input_urdf_dir, 'mobility.urdf')
    output_path = os.path.join(output_dir, 'mobility.urdf')

    shutil.copytree(input_urdf_dir, output_dir, dirs_exist_ok=True)  # Ensure directory exists
    os.remove(output_path)  # Remove old URDF file if it exists

    semantic_path = os.path.join(output_dir, 'semantics.txt')
    with open(semantic_path, "r") as f:
        orig_semantics = f.readlines()

    print(orig_semantics)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect all revolute joints with links to remove
    revolute_joints_to_remove = []
    links_to_remove = set()

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        child_link = joint.find("child").get("link")
        # print(joint, joint_type, child_link, joint.find("axis"), joint_type == "fixed", joint.find("axis") is not None)
        if joint_type == "revolute" and child_link not in link_list:
            revolute_joints_to_remove.append(joint)
            links_to_remove.add(child_link)
        # Handle fixed joints that have an axis (they were originally movable)
        elif joint_type != "fixed" and joint.find("axis") is not None:
            # print(child_link, link_list)
            if child_link not in link_list:
                # print("REMOVE", joint, child_link)
                revolute_joints_to_remove.append(joint)
                links_to_remove.add(child_link)
            

    if len(links_to_remove) != 0:
        # Remove extra revolute joints
        for joint in revolute_joints_to_remove:
            root.remove(joint)

        # Remove associated links
        for link in root.findall("link"):
            if link.get("name") in links_to_remove:
                root.remove(link)


    # Keep only lines that don't contain the keyword
    filtered_semantics = []
    for semantics in orig_semantics:
        kept = True
        for remove_link in links_to_remove:
            print(remove_link, semantics)
            if remove_link in semantics:
                kept=False
                break
        
        if kept:
            filtered_semantics.append(semantics)

    # Write the modified content back
    with open(semantic_path, "w") as f:
        f.writelines(filtered_semantics)

    tree.write(output_path)
    return len(links_to_remove) != 0  # Have no handle copy
    # return [joint.get("name") for joint in revolute_joints_to_remove], list(links_to_remove)


def create_open_video(raw_id, target_link, pm_dir="/data/datasets/failure_history_door/raw/"):
    # obj_id = "8867_3"
    pm_dir = os.path.join(os.path.expanduser(pm_dir), "raw")
    obj_ids = [name for name in os.listdir(pm_dir) if raw_id in name and target_link in name]
    for obj_id in obj_ids:
        print(obj_id)
        raw_data = PMObject(os.path.join(pm_dir, obj_id))

        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]
        # print(available_joints)

        try:
            env = PMSuctionSim(obj_id, pm_dir, gui=False)
        except:
            breakpoint()
        env.reset()
        for joint in available_joints:
            info = p.getJointInfo(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[joint],
                env.render_env.client_id,
            )
            init_angle, target_angle = info[8], info[9]
            env.set_joint_state(joint, init_angle)

        # Get the init & target angle for the target link
        info = p.getJointInfo(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[target_link],
            env.render_env.client_id,
        )
        init_angle, target_angle = info[8], info[9]

        # Camera param
        frame_width = 640
        frame_height = 480
        writer = imageio.get_writer(os.path.join(pm_dir, obj_id, "open.mp4"), fps=5)
        print("writing to ", os.path.join(pm_dir, obj_id, "open.mp4"))
        env.set_writer(writer)

        angle_sample_cnts = 20
        for angle_ratio, angle in zip(
                np.linspace(0, 100, angle_sample_cnts),
                np.linspace(init_angle, target_angle, angle_sample_cnts)
            ):
            env.set_joint_state(target_link, angle)

            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs


            env.gripper.set_velocity([0, 0, 0], [0, 0, 0])
            for i in range(10):
                p.stepSimulation(env.render_env.client_id)
            # print(angle)

            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=frame_width,
                height=frame_height,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=5,
                    # yaw=180,
                    yaw=270,
                    # pitch=90,
                    pitch=-30,
                    roll=0,
                    upAxisIndex=2,
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=float(frame_width) / frame_height,
                    nearVal=0.1,
                    farVal=100.0,
                ),
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=env.render_env.client_id,
            )
            image = np.array(rgbImg, dtype=np.uint8)
            image = image[:, :, :3]

            # imageio.imwrite(f'./image_{int(angle_ratio * 100)}.jpg',rgb[:, :, :3])

            # Add the frame to the video
            writer.append_data(image)
            # writer.append_data(rgb[:, :, :3])

        writer.close()

def arg_parser():
    parser = argparse.ArgumentParser(description='Generate URDF files with for Multimodal Doors.')
    parser.add_argument('--pm_path', type=str, default= "/home/yishu/datasets/partnet-mobility", help='The path to partnet mobility dataset.')
    parser.add_argument('--output_md_path', type=str, default= "/home/yishu/datasets/failure_history_door", help='The path to the output multimodal door dataset.')
    parser.add_argument('--save_video', type=bool, default=True)
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    # All partnet mobility door ids
    door_dataset = ["8877",
        "8893",
        "8897",
        "8903",
        "8919",
        "8930",
        "8961",
        "8997",
        "9016",
        "9035",
        "9041",
        "9065",
        "9070",
        "9107",
        "9117",
        "9127",
        "9128",
        "9148",
        "9164",
        "9168",
        "9277",
        "9280",
        "9281",
        "9288",
        "9386",
        "9388",
        "9410",
        "8867", "8983", "8994", "9003", "9263", "9393"]
    # 2 sided doors
    two_doors = ["8877", "8919", "8930", "8961", "8997", "9016", "9128", "9168", "9280"]
    
    with open('./metadata/movable_links_fullset_000_full.json', 'r') as f:
        movable_links = json.load(f)
    
    for obj_name in tqdm(door_dataset):
        for link_id in  movable_links[obj_name]:   
            obj_id = f'{obj_name}_{link_id}'
            have_handle = create_no_handle_replicate(f"{args.pm_path}/raw/{obj_name}/", movable_links[obj_name], f"{args.output_md_path}/raw/{obj_id}_1/")
            if have_handle:
                modify_urdf(f"{args.pm_path}/raw/{obj_name}/", f"{args.output_md_path}/raw", obj_id, link_id, handle=True)
            modify_urdf(f"{args.output_md_path}/raw/{obj_id}_1/", f"{args.output_md_path}/raw", obj_id, link_id, handle=False, axis_change=obj_name not in two_doors, skip_first=True)  # If there's no doors, even after removing handles, only create push & pull ambiguity

            create_open_video(obj_name, link_id, args.output_md_path)


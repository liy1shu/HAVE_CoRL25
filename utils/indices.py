import os
import json
import pickle

folder_path = '<path to the dataset folder>'

results = {
            "data_idx": [],
            "traj_idx": [],
            "action_idx": []
        }

for file_name in os.listdir(folder_path):
    if file_name.endswith('.pkl'):
        file_path = os.path.join(folder_path, file_name)
        
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        continue

    for i, item in enumerate(data):
        for idx in range(len(item['evaluate']['scores'])):
            results['data_idx'].append(data[i]['obj_id'])
            results['traj_idx'].append(i)
            results['action_idx'].append(idx)
    
    if (len(data) != 20):
        print(f'Data {file_name} has only {len(data)} trajs')

# save the indices as a json file
with open(os.path.join(folder_path, 'indices.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("JSON file has been created successfully.")

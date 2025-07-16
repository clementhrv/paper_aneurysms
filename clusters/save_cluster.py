import numpy as np
import os
import pandas as pd
import trimesh

df = pd.read_csv(r'C:\Users\terve\Documents\0-MINES\Projet_TRELLIS\clusters\clusters_tsne_mean_feat.csv')
df = df.iloc[:, :-1]
df_sorted = df.sort_values(by='cluster')
obj_folder = r'C:\Users\terve\Documents\0-MINES\Projet_TRELLIS\clusters\surface_dataset'

def save_one_cluster(num_cluster: int, df_sorted, output_folder, sample_size=2000):
    cluster_data = df_sorted[df_sorted['cluster'] == num_cluster]
    len_cluster = len(cluster_data)
    i_max = int(np.sqrt(len_cluster))
    count = 0

    meshes = []

    for filename in cluster_data['filename']:
        i = count // i_max
        j = count % i_max
        count += 1

        obj_filename = filename.replace('.npz', '.obj')
        obj_path = os.path.join(obj_folder, obj_filename)
        
        if os.path.exists(obj_path):
            mesh = trimesh.load(obj_path)
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Invalid mesh for {obj_path}, skipping.")
                continue  # Skip if not a proper mesh

            mask = mesh.vertices[:, 1] >= 5
            if np.any(mask):  
                print(f"Processing {obj_path} with {np.sum(mask)} valid vertices.")
                mesh.vertices[:, 2] += 10 * i
                mesh.vertices[:, 0] += 20 * j
                meshes.append(mesh)

    print(f"Cluster {num_cluster} contains {len(meshes)} meshes.")

    if meshes:
        print(f"Saving cluster {num_cluster} with {len(meshes)} meshes.")
        combined = trimesh.util.concatenate(meshes)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'cluster_{num_cluster}_combined.obj')
        combined.export(output_path)


for i in range(df_sorted['cluster'].nunique()):
    save_one_cluster(i, df_sorted, r'C:\Users\terve\Documents\0-MINES\Projet_TRELLIS\clusters\output_3')

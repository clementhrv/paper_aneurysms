import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import trimesh

# Load your data
df = pd.read_csv(r'C:\Users\terve\Documents\0-MINES\Projet_TRELLIS\clusters\clusters_tsne_mean.csv')
df = df.iloc[:, :-1]
df_sorted = df.sort_values(by='cluster')
obj_folder = r'C:\Users\terve\Documents\0-MINES\Projet_TRELLIS\clusters\surface_dataset'

def show_one_cluster(num_cluster: int, df_sorted, title="Clusters", sample_size=2000):
    fig = plt.figure(figsize=(42, 28), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    cluster_data = df_sorted[df_sorted['cluster'] == num_cluster]
    len_cluster = len(cluster_data)
    i_max = int(np.sqrt(len_cluster))
    count = 0

    for filename in cluster_data['filename']:
        i = count // i_max
        j = count % i_max
        count += 1
        obj_filename = filename.replace('.npz', '.obj')
        obj_path = os.path.join(obj_folder, obj_filename)
        if os.path.exists(obj_path):
            mesh = trimesh.load(obj_path)
            vertices = mesh.vertices

            vertices = vertices[vertices[:, 1] >= 5]

            if len(vertices) > sample_size:
                indices = np.random.choice(len(vertices), size=sample_size, replace=False)
                vertices = vertices[indices]

            shifted_vertices = vertices.copy()
            shifted_vertices[:, 2] += 10 * i
            shifted_vertices[:, 0] += 20 * j

            base_color = plt.cm.tab10(num_cluster % 10)
            edge_color = tuple([max(c - 0.3, 0) for c in base_color[:3]] + [1])

            ax.scatter(
                shifted_vertices[:, 2], 
                shifted_vertices[:, 0], 
                shifted_vertices[:, 1],
                s=45 * (fig.dpi / 72)**2,  # DPI-scaled size
                color=base_color,
                alpha=0.9,
                edgecolors=edge_color,
                linewidth=0.3,
                marker='o'
            )
        else:
            print(f"Fichier non trouvé : {obj_path}")

    ax.set_box_aspect([max(i_max//3, 1), i_max, 1])
    ax.set_zlim(top=16)
    ax.view_init(elev=25, azim=135)

    # Clean, custom grid setup
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # transparent panes
        axis.pane.set_edgecolor('lightgray')
        axis.pane.set_linewidth(0.8)

    xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5)
    yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5)
    zticks = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 4)

    for x in xticks:
        ax.plot([x, x], [yticks[0], yticks[-1]], [zticks[0]]*2, linestyle='--', color='lightgray', linewidth=0.5, alpha=0.3)

    for y in yticks:
        ax.plot([xticks[0], xticks[-1]], [y, y], [zticks[0]]*2, linestyle='--', color='lightgray', linewidth=0.5, alpha=0.3)

    for z in zticks:
        ax.plot([xticks[0], xticks[-1]], [yticks[0]]*2, [z, z], linestyle='--', color='lightgray', linewidth=0.5, alpha=0.3)

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=8, direction='in', pad=10)

    ax.set_position([0.05, 0.1, 0.92, 0.85])  # Left, bottom, width, height

    fig.savefig(f"cluster_{num_cluster}_plot.png", dpi='figure', bbox_inches='tight')
    # plt.show()

# Example usage
# show_one_cluster(0, df_sorted, title="Cluster 0 - Example")

for i in range(14):
    show_one_cluster(i, df_sorted, title="Cluster {}".format(i))


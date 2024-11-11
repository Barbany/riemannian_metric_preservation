import argparse
import os
import time

import numpy as np
import pymeshlab
from tqdm import tqdm


def main(data_root):
    for seq in tqdm(os.listdir(data_root)):
        vertex_file = os.path.join(data_root, seq, "model", "mesh_vertices.tsv")
        if os.path.isfile(vertex_file):
            vertices = np.loadtxt(
                vertex_file,
                delimiter="\t",
            )
            faces = (
                np.loadtxt(
                    os.path.join(data_root, seq, "model", "mesh_faces.tsv"),
                    delimiter="\t",
                    dtype=int,
                )
                + 1
            )
            template_mesh_name = os.path.join(data_root, seq, "model", "template.obj")
            with open(template_mesh_name, "w") as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(template_mesh_name)
            ms.meshing_re_orient_faces_coherently()
            ms.meshing_invert_face_orientation()
            ms.compute_texcoord_parametrization_least_squares_conformal_maps()
            ms.save_current_mesh(template_mesh_name)
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    main(args.data_root)

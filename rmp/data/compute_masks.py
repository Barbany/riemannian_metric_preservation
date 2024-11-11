import argparse
import os

import numpy as np
from PIL import Image
from skimage.morphology import convex_hull_image
from tqdm import tqdm


def main(data_root):
    for seq in tqdm(os.listdir(data_root)):
        data_path = os.path.join(data_root, seq)

        os.makedirs(os.path.join(data_path, "masks"), exist_ok=True)

        K = np.loadtxt(os.path.join(data_path, "model/cam_intrinsic.tsv"), delimiter="\t")

        for img_file in tqdm(
            os.listdir(os.path.join(data_path, "seq")),
            desc=f"Sequence: {seq}",
        ):
            img = Image.open(os.path.join(data_path, "seq", img_file))
            vertices = np.loadtxt(
                os.path.join(data_path, "ground_truth", img_file.replace(".png", ".tsv"))
            )
            uv_unnorm = K @ vertices.T
            vertices_image = (uv_unnorm[:-1] / uv_unnorm[-1]).T
            mask = np.zeros((img.size[1], img.size[0]))
            coords = np.minimum(
                np.maximum(np.round(vertices_image).astype(int), 0),
                np.array(img.size) - 1,
            )
            mask[coords[:, 1], coords[:, 0]] = 1
            chull = convex_hull_image(mask)
            Image.fromarray((chull * 255).astype(np.uint8)).save(
                os.path.join(data_path, "masks", img_file)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    main(args.data_root)

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def main(data_root):
    padding = 10
    for seq in tqdm(os.listdir(data_root)):
        data_path = os.path.join(data_root, seq)

        os.makedirs(os.path.join(data_path, "cropped_seq"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "cropped_masks"), exist_ok=True)

        ymin = ymax = xmin = xmax = None
        for img_file in tqdm(os.listdir(os.path.join(data_path, "masks"))):
            mask_array = np.asarray(Image.open(os.path.join(data_path, "masks", img_file)))
            # Find where the white pixels are
            rows = np.any(mask_array, axis=1)
            cols = np.any(mask_array, axis=0)

            # Find the bounding box coordinates
            curr_ymin, curr_ymax = np.where(rows)[0][[0, -1]]
            curr_xmin, curr_xmax = np.where(cols)[0][[0, -1]]
            if ymin is None:
                ymin = curr_ymin
                ymax = curr_ymax
                xmin = curr_xmin
                xmax = curr_xmax
            else:
                ymin = min(curr_ymin, ymin)
                xmin = min(curr_xmin, xmin)

                ymax = max(curr_ymax, ymax)
                xmax = max(curr_xmax, xmax)

        # Add padding
        assert ymin is not None and ymax is not None
        assert xmin is not None and xmax is not None
        ymin = max(ymin - padding, 0)
        ymax = min(ymax + padding, mask_array.shape[0])
        xmin = max(xmin - padding, 0)
        xmax = min(xmax + padding, mask_array.shape[1])

        K = np.loadtxt(
            os.path.join(data_path, "model", "cam_intrinsic.tsv"),
            delimiter="\t",
        )
        K_cropped = K.copy()
        K_cropped[0, -1] -= xmin
        K_cropped[1, -1] -= ymin

        np.savetxt(
            os.path.join(data_path, "model", "cropped_cam_intrinsic.tsv"),
            K_cropped,
            delimiter="\t",
        )

        for img_file in tqdm(
            os.listdir(os.path.join(data_path, "seq")),
            desc=f"Sequence: {seq}",
        ):
            mask = Image.open(os.path.join(data_path, "masks", img_file))
            img = Image.open(os.path.join(data_path, "seq", img_file))

            mask.crop((xmin, ymin, xmax, ymax)).save(
                os.path.join(data_path, "cropped_masks", img_file)
            )
            img.crop((xmin, ymin, xmax, ymax)).save(
                os.path.join(data_path, "cropped_seq", img_file)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    main(args.data_root)

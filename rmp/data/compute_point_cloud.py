import argparse
import os

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


def main(args):
    for seq in tqdm(os.listdir(args.data_root)):
        image_folder = "cropped_seq" if args.cropped else "seq"

        pc_path = os.path.join(
            args.data_root,
            seq,
            "point_cloud",
        )
        os.makedirs(pc_path, exist_ok=True)

        K = np.loadtxt(
            os.path.join(
                args.data_root,
                seq,
                "model",
                "cropped_cam_intrinsic.tsv" if args.cropped else "cam_intrinsic.tsv",
            ),
            delimiter="\t",
        )

        if os.path.isdir(os.path.join(args.data_root, seq, image_folder)):
            for img in tqdm(
                os.listdir(os.path.join(args.data_root, seq, image_folder)),
                desc=f"Sequence {seq}",
            ):
                pil_img = Image.open(os.path.join(args.data_root, seq, image_folder, img)).convert(
                    "RGB"
                )
                image = o3d.geometry.Image(np.array(pil_img))
                depth = o3d.geometry.Image(
                    np.load(
                        os.path.join(
                            args.data_root,
                            seq,
                            args.depth_folder,
                            img.replace(".png", ".npy"),
                        )
                    )
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    image, depth, depth_scale=1, convert_rgb_to_intensity=False
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd,
                    intrinsic=o3d.camera.PinholeCameraIntrinsic(pil_img.width, pil_img.height, K),
                )
                o3d.io.write_point_cloud(os.path.join(pc_path, img.replace(".png", ".ply")), pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--depth_folder", type=str, default="cropped_depth_zoe_np")
    parser.add_argument("--use_uncropped", dest="cropped", action="store_false")
    args = parser.parse_args()
    main(args)

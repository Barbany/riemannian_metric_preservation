import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def main(args):
    torch.hub.help(
        "intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True
    )  # Triggers fresh download of MiDaS repo

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    assert isinstance(model_zoe_nk, torch.nn.Module)
    zoe = model_zoe_nk.to("cuda")

    for seq in tqdm(os.listdir(args.data_root)):
        image_folder = "cropped_seq" if args.cropped else "seq"

        np_depth_path = os.path.join(
            args.data_root,
            seq,
            "cropped_depth_zoe_np" if args.cropped else "depth_zoe_np",
        )
        pil_depth_path = os.path.join(
            args.data_root,
            seq,
            "cropped_depth_zoe_pil" if args.cropped else "depth_zoe_pil",
        )
        os.makedirs(np_depth_path, exist_ok=True)
        os.makedirs(pil_depth_path, exist_ok=True)
        if os.path.isdir(os.path.join(args.data_root, seq, image_folder)):
            for img in tqdm(
                os.listdir(os.path.join(args.data_root, seq, image_folder)),
                desc=f"Sequence {seq}",
            ):
                image = Image.open(os.path.join(args.data_root, seq, image_folder, img)).convert(
                    "RGB"
                )
                depth_numpy = zoe.infer_pil(image)
                np.save(
                    os.path.join(
                        np_depth_path,
                        img.replace(".png", ".npy"),
                    ),
                    depth_numpy,
                )
                depth_pil = zoe.infer_pil(image, output_type="pil")
                depth_pil.save(os.path.join(pil_depth_path, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--use_uncropped", dest="cropped", action="store_false")
    args = parser.parse_args()
    main(args)

import argparse
import os

import numpy as np
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize
from natsort import natsorted
from PIL import Image, ImageDraw
from tqdm import tqdm


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def main(args):
    for seq in tqdm(os.listdir(args.data_root)):
        data_path = os.path.join(args.data_root, seq)
        tracks = np.array([
            np.loadtxt(os.path.join(data_path, args.matches, file_name))[:, 6:8]
            for file_name in natsorted(os.listdir(os.path.join(data_path, args.matches)))
        ])
        visibility = np.array([
            np.loadtxt(os.path.join(data_path, args.matches, file_name))[:, -1]
            for file_name in natsorted(os.listdir(os.path.join(data_path, args.matches)))
        ])[None]

        out_path = os.path.join(data_path, args.matches + "_viz")
        os.makedirs(out_path, exist_ok=True)

        img_dir = "cropped_seq" if "cropped" in args.matches else "seq"

        res_video = [
            np.asarray(Image.open(os.path.join(data_path, img_dir, file_name)))
            for file_name in natsorted(os.listdir(os.path.join(data_path, img_dir)))
        ]
        T = len(res_video)
        N = tracks.shape[1]

        vector_colors = np.zeros((T, N, 3))

        query_frame = 0
        compensate_for_camera_motion = False

        y_min, y_max = (
            tracks[query_frame, :, 1].min(),
            tracks[query_frame, :, 1].max(),
        )
        norm = Normalize(y_min, y_max)
        for n in range(N):
            color = cm.get_cmap("gist_rainbow")(norm(tracks[query_frame, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)

        #  draw points
        for t in range(query_frame, T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion:
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(4),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                        )
            img.save(os.path.join(out_path, f"{t:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--matches", type=str, required=True)
    args = parser.parse_args()
    main(args)

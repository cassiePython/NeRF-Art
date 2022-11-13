# %%
import numpy as np
from read_wrote_model import read_images_text, qvec2rotmat, read_cameras_text, read_cameras_binary, read_images_binary
import argparse
from utils import io_util
# please change the path to your own path
parser = io_util.create_args_parser()
parser.add_argument('--root_dir', type=str,
                    default="./data/fangzhou_mouth/")

args, _ = parser.parse_known_args()

root_dir = args.root_dir

cameras = read_cameras_binary(root_dir+"sparse/0/cameras.bin")

images = read_images_binary(root_dir+"sparse/0/images.bin")
K = np.eye(3)
K[0, 0] = cameras[1].params[0]
K[1, 1] = cameras[1].params[1]
K[0, 2] = cameras[1].params[2]
K[1, 2] = cameras[1].params[3]

cameras_npz_format = {}

for ii in range(len(images)):
    # for ii in range(96):
    cur_image = images[ii+1]

    M = np.zeros((3, 4))
    M[:, 3] = cur_image.tvec
    M[:3, :3] = qvec2rotmat(cur_image.qvec)

    P = np.eye(4)
    P[:3, :] = K@M
    cameras_npz_format['world_mat_%d' % ii] = P

np.savez(root_dir +
         "cameras_before_normalization.npz",
         **cameras_npz_format)
print("Done")

# %%

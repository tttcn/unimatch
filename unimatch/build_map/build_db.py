import configparser
import logging
import pickle
import sys
import time

import cv2
import nmslib
import numpy as np

from unimatch.map import MapPose
from unimatch.utils import Feature

logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = "../config/default.ini"

config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH, "utf-8")

# 设定相机参数，这里的参数在pnp中要用到, 初始化特征座标的时候也要用到。
FX = config.getfloat('camera', 'FX')
FY = config.getfloat('camera', 'FY')
CX = config.getfloat('camera', 'CX')
CY = config.getfloat('camera', 'CY')
K1 = config.getfloat('camera', 'K1')
K2 = config.getfloat('camera', 'K2')
P1 = config.getfloat('camera', 'P1')
P2 = config.getfloat('camera', 'P2')
CAMERA_FACTOR = config.getfloat('camera', 'CAMERA_FACTOR')

CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]],
                         dtype=np.double)
DIST_COEFFS = np.array([K1, K2, P1, P2], dtype=np.double)

# 设定hnsw参数
M = config.getint("hnsw", "M")
EF_C = config.getint("hnsw", "EF_C")
NUM_THREADS = config.getint("hnsw", "NUM_THREADS")

SPACE_NAME = config.get("hnsw", "SPACE_NAME")


def pix2cam(point, depth_img):
    """
    this function calculates the 3d coordinate of a 2d point in a picture,
    the 3d coordinate is
    """
    # 如果depth_img和rgb_map尺寸不一样的话, 需要做比例变换, 尺寸一致就没有必要了
    uf, vf = point

    ul = np.floor(uf)
    ug = np.ceil(uf)
    vl = np.floor(vf)
    vg = np.ceil(vf)

    uli = int(ul)
    ugi = int(ug)
    vli = int(vf)
    vgi = int(vf)

    # opencv convention: (u,v)->depth_img[v][u] u是width v是height
    d = ((ug - uf) * (vg - vf) * np.double(depth_img[vgi][ugi]) + (ug - uf) *
         (vf - vl) * np.double(depth_img[vli][ugi]) + (uf - ul) *
         (vg - vf) * np.double(depth_img[vgi][uli]) + (uf - ul) *
         (vf - vl) * np.double(depth_img[vli][uli]))

    # d455 在400mm内是不可靠的, 建议的精确范围应该是4m-12m
    if d < 400.0:
        return False, np.zeros((3, 1))
    z = d / CAMERA_FACTOR
    x = (uf - CX) * z / FX
    y = (vf - CY) * z / FY
    local_xyz = np.mat([x, y, z]).T
    return True, local_xyz


sift_extractor = cv2.SIFT_create()


def sift_extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift_extractor.detectAndCompute(gray, None)
    return kp, des


class ColmapImageInfo:
    """
    This class is used to process the aligned colmap model (text file)
    """

    def __init__(self, data_line: str):
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        data_segs = data_line.rstrip().split(" ")  # rstrip remove /t/n (CRLF)
        self.img_id = data_segs[9]  # NAME
        self.qvec = np.array([float(data_segs[i]) for i in range(1, 5)])  # wxyz
        self.tvec = np.array([float(data_segs[i]) for i in range(5, 8)])
        # pytransform3d (x,y,z,qw,qx,qy,qz)
        self.pose = np.hstack((self.tvec, self.qvec))
        self.colmap_id = int(data_segs[0])
        self.map_pose = MapPose(self.pose)


def gen_db_map_poses(db_map_poses_path: str, db_col_path: str) -> dict:
    """
    this function reads aligned colmap model from a text file,
    and dump a binary file contains the pose information.
    """
    db_cii_list = []

    with open(db_col_path) as col_file:
        for line_idx, line in enumerate(col_file):
            if line_idx > 3 and line_idx % 2 == 0:
                cii = ColmapImageInfo(line)
                db_cii_list.append(cii)

    db_map_poses = dict()
    for cii in db_cii_list:
        print(cii.img_id)
        db_map_poses[cii.img_id] = cii.map_pose

    with open(db_map_poses_path, "wb") as file:
        pickle.dump(db_map_poses, file)

    return db_map_poses


def gen_db_features(
        db_features_path,
        db_map_poses,
        rgb_dir,
        depth_dir,
) -> list:
    """
    extract SIFT features from database images and dump a Feature array.
    only use features from images that have MapPose, and features has valid depth.
    """
    db_sift_num = 0
    db_features = []
    feature_id = 0

    for img_id in db_map_poses.keys():
        if img_id not in db_map_poses:  # no contribution to final pnp
            continue

        rgb_path = rgb_dir + img_id
        depth_path = depth_dir + img_id[:-5] + "d.png"
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, -1)

        kp, des = sift_extract(rgb)
        feature_num = len(kp)
        db_sift_num += feature_num
        for i in range(feature_num):
            d = des[i]
            k = kp[i]
            xyz_is_exist, xyz = pix2cam(k.pt, depth)
            if xyz_is_exist:
                db_features.append(
                    Feature(img_id, feature_id, d.astype(np.uint8), k, True, xyz)
                )
            feature_id += 1

    logger.info("all 2d sift features:", db_sift_num)
    logger.info("valid sift features with depth:", len(db_features))

    with open(db_features_path, "wb") as file:
        pickle.dump(db_features, file)
    return db_features


def gen_db(db_path, db_features):
    """
    use all valid features in db_features to build a hnsw database
    """
    fm = np.array(
        [feature.des for feature in db_features if feature.depth_is_valid],
        dtype=np.uint8,
    )
    print(len(fm))
    # Set index parameters
    # These are the most important onese

    index_time_params = {
        "M": M,
        "indexThreadQty": NUM_THREADS,
        "efConstruction": EF_C,
        "post": 0,
        "skip_optimized_index": 0,  # using non-optimized index! !!!
    }

    # Initialize the library, specify the space, the type of the vector and add data points
    # for SIFT data, we want DENSE_UINT8_VECTOR and distance type INT
    db = nmslib.init(
        method="hnsw",
        space=SPACE_NAME,
        data_type=nmslib.DataType.DENSE_UINT8_VECTOR,
        dtype=nmslib.DistType.INT,
    )
    db.addDataPointBatch(fm)
    start = time.time()
    db.createIndex(index_time_params)
    end = time.time()
    logger.info("Index-time parameters", index_time_params)
    logger.info("Indexing time = %f" % (end - start))
    db.saveIndex(db_path, save_data=1)


# change the path
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: build_db.py database_directory aligned_col_path rgb_images_directory depth_images_directory"
        )
        print("absolute path is recommended, directory should end with /")
        exit()

    database_directory = sys.argv[1]
    aligned_col_path = sys.argv[2]
    rgb_images_directory = sys.argv[3]
    depth_images_directory = sys.argv[4]

    # database map poses
    db_map_poses = gen_db_map_poses(
        database_directory + "db_map_poses.bin",
        aligned_col_path,
    )
    # database features
    db_features = gen_db_features(
        database_directory + "db_features.bin",
        db_map_poses,
        rgb_images_directory,
        depth_images_directory,
    )
    # with open('/home/tao/dataset/office/unimatch_database/db_features.bin',
    #           'rb') as file:
    #     dbf = pickle.load(file)
    gen_db(database_directory + "db_index.bin", db_features)

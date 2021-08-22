import configparser
import logging
import pickle

import cv2
import nmslib
import numpy as np
from pytransform3d import transformations as pt
from scipy.spatial.transform import Rotation as Rotation

from unimatch.map import MapPose

# TODO: 并行处理pnp
# TODO: configparser取消全局配置
# TODO: 重构代码，使之兼容orb feature，或者多种其他feature


CONFIG_FILE_PATH = 'config/default.ini'

config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH, 'utf-8')

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
M = config.getint('hnsw', 'M')
EF_C = config.getint('hnsw', 'EF_C')
NUM_THREADS = config.getint('hnsw', 'NUM_THREADS')

SPACE_NAME = config.get('hnsw', 'SPACE_NAME')

# 设定查询参数
EF_S = config.getint('hnsw', 'EF_S')
QUERY_TIME_PARAMS = {"efSearch": EF_S}

# 设定unimatch参数
MIN_HIT_BAR = config.getint('unimatch', 'MIN_HIT_BAR')
SEARCH_TOPK = config.getint('unimatch', 'SEARCH_TOPK')
TOP_HIT = config.getint('unimatch', 'TOP_HIT')


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

class UniMatchService(object):
    """
    this class provide the UniMatch algorithm service
    """

    def __init__(self, db_path, db_features_path, db_map_poses_path) -> None:
        self.logger = logging.getLogger(__name__)
        # 设定hnsw参数
        self.num_threads = NUM_THREADS
        self.search_topk = SEARCH_TOPK
        self.top_hit = TOP_HIT

        # init db
        # Initialize the library, specify the space, the type of the vector
        # and add data points for SIFT data, we want DENSE_UINT8_VECTOR
        # and distance type INT
        self.db = nmslib.init(
            method="hnsw",
            space="l2sqr_sift",
            data_type=nmslib.DataType.DENSE_UINT8_VECTOR,
            dtype=nmslib.DistType.INT,
        )
        self.logger.info("Loading db from", db_path)
        self.db.loadIndex(db_path, load_data=1)
        self.logger.info("Setting query-time parameters", QUERY_TIME_PARAMS)
        self.db.setQueryTimeParams(QUERY_TIME_PARAMS)

        self.db_features = None
        with open(db_features_path, "rb") as file:
            self.db_features = pickle.load(file)
        self.db_map_poses = None
        with open(db_map_poses_path, "rb") as file:
            self.db_map_poses = pickle.load(file)

        self.sift_extractor = cv2.SIFT_create()

        # might be useful in next version
        # self.last_xyz = None
        # self.speed = (0, 0, 0)

        self.logger.info("unimatch service is ready")

    def sift_extract(self, img):
        """
        extract SIFT features from img
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift_extractor.detectAndCompute(gray, None)
        return kp, des

    def search(self, qkp, qdes):
        """
        search some candidate reference images that are similar to the query
        """
        nbrs = self.db.knnQueryBatch(
            np.array(qdes, dtype=np.uint8),
            k=self.search_topk,
            num_threads=self.num_threads,
        )

        # 召回
        hit_dic = dict()  # hit_dic[img_id]=hit on img
        match_dic = dict()  # match_dic[img_id]=[(query_feature_id,feature_id)]
        query_feature_idx = 0

        for feature_ids, distances in nbrs:
            for feature_id, dis in zip(feature_ids, distances):
                # might not be necessary
                if not self.db_features[feature_id].depth_is_valid:
                    continue
                img_id = self.db_features[feature_id].img_id
                if img_id not in hit_dic:
                    hit_dic[img_id] = 1
                    match_dic[img_id] = [(query_feature_idx, feature_id)]
                else:
                    hit_dic[img_id] += 1
                    match_dic[img_id].append((query_feature_idx, feature_id))
            query_feature_idx += 1

        # top_hit images
        img_ids = [
            hit_pair[0] for hit_pair in sorted(
                hit_dic.items(), key=lambda item: item[1])[-self.top_hit:]
        ]

        # pick top hit images to as final match result
        img_dic = dict()
        for img_id in img_ids:
            # exclude some cases when the similarity (hit_num) is too low
            if hit_dic[img_id] > MIN_HIT_BAR:
                img_dic[img_id] = match_dic[img_id]
        return img_dic

    #
    def single_img_pnp(self, img_id, matches, qkp, qdes):
        """
        query image pnp with single db image, return a relative pose_mat
        """
        objPts = []
        imgPts = []
        for i in range(len(matches)):
            db_feature = self.db_features[matches[i][1]]
            if db_feature.depth_is_valid is True:
                objPts.append(db_feature.local_xyz)
                imgPts.append(qkp[matches[i][0]].pt)

        pose_mat = np.identity(4)
        pose_mat[3][3] = 1.0

        pnp_is_success = True

        try:
            result, rvec, tvec, inliners = cv2.solvePnPRansac(
                np.array(objPts),
                np.array(imgPts),
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            mt = np.array(tvec.T)[0]
            mR = Rotation.from_rotvec(rvec.T)
            pose_mat = pt.transform_from(mR.as_matrix(), mt)
            self.logger.info("inliners number: {}", len(inliners))
        except Exception:
            inliners = [(1)]
            pose_mat = np.identity(4)
            pose_mat[3][3] = 1.0
            pnp_is_success = False
            self.logger.warning("final to pnp")

        return pnp_is_success, img_id, pose_mat, len(inliners)

    def estimate_pose(self, relative_mat, db_img_id) -> MapPose:
        db_world2cam_mat = self.db_map_poses[db_img_id].world2cam_mat
        cal_world2cam_mat = relative_mat @ db_world2cam_mat
        cal_world2cam_pose = pt.pq_from_transform(cal_world2cam_mat)
        pose = MapPose(cal_world2cam_pose)
        return pose

    def get_pose(self, query_img):
        qkp, qdes = self.sift_extract(query_img)
        candidate_dict = self.search(qkp, qdes)
        max_inliners = 0
        best_db2query_mat = None
        db_img_id = ''

        get_pose_success = False

        # serial pnp
        for img_id, matches in candidate_dict.items():
            is_success, img_id, db2query_mat, len_in = self.single_img_pnp(
                img_id, matches, qkp, qdes)
            if is_success:
                get_pose_success = True
                if max_inliners < len_in:
                    max_inliners = len_in
                    best_db2query_mat = db2query_mat
                    db_img_id = img_id

        final_pose = MapPose([0, 0, 0, 0, 0, 0, 0])

        if get_pose_success:
            final_pose = self.estimate_pose(best_db2query_mat, db_img_id)
            self.logger.info("")
        return get_pose_success, final_pose

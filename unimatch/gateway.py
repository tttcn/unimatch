import sys
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from unimatch.core import UniMatchService

sys.path.append("..")

app = FastAPI()
unimatch_service = UniMatchService(
    "database/db_index.bin",
    "database/db_features.bin",
    "database/db_map_poses.bin",
)


# 正式接口
@app.post("/position")
async def get_position(image_file: bytes = File(...)):
    buf = np.frombuffer(image_file, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
    is_success, pose = unimatch_service.get_pose(image)
    print(pose.xyz, type(pose.xyz))
    if is_success:
        response = {
            "status": "Positioning is success",
            "xyz": pose.xyz.tolist(),
            "direction": [pose.c.real, pose.c.imag]
        }
    else:
        response = {
            "status": "Positioning is failed",
        }
    return response


# 调试接口
@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    print("get?")
    file = files[0]
    buf = np.frombuffer(file, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
    is_success, pose = unimatch_service.get_pose(image)
    print(pose.xyz, type(pose.xyz))
    if is_success:
        response = {
            "status": "Positioning is success",
            "xyz": pose.xyz.tolist(),
            "direction": [pose.c.real, pose.c.imag]
        }
    else:
        response = {
            "status": "Positioning is failed",
        }
    return response


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    print("get?")
    return {
        "filenames": [file.filename for file in files],
        "file_sizes": [len(file.file) for file in files],
    }


# 调试界面
@app.get("/")
async def main():
    content = """
        <body>
        <head>unitmatch test</head>
        <form action="/files/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </body>
    """
    return HTMLResponse(content=content)

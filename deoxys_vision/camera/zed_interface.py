import json
import threading
from typing import Any, Optional

import pyzed.sl as sl
import numpy as np
from easydict import EasyDict
from deoxys_vision.threading.threading_utils import Worker
import cv2


def get_zed_intrinsics_param(K_matrix: np.ndarray):
    return {"fx": K_matrix[0][0], "fy": K_matrix[1][1], "cx": K_matrix[0][2], "cy": K_matrix[1][2]}


class ZEDWorker(Worker):
    def __init__(self, device_id, resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.PERFORMANCE):
        self.device_id = device_id
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.depth_mode = depth_mode
        # self.init_params.sdk_verbose = 1
        # self.init_params.camera_image_flip = sl.FLIP_MODE.OFF
        self.last_obs = None
        self.init_params.camera_fps = 60
        self.init_params.set_from_camera_id(self.device_id)
        self.runtime_params = sl.RuntimeParameters()
        self.calibration = {
            "color": {"intrinsics": None, "distortion": None},
            "depth": {"intrinsics": None, "distortion": None},
        }
        super().__init__()

    def run(self) -> None:
        if self.camera.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            raise Exception("Unable to open ZED camera")
        calibration_params = self.camera.get_camera_information().calibration_parameters.left_cam
        intrinsics = np.array([
            [calibration_params.fx, 0, calibration_params.cx],
            [0, calibration_params.fy, calibration_params.cy],
            [0, 0, 1]
        ])
        distortion = np.array(calibration_params.disto)
        # According to zed documentation depth camera and left rgb camera are algigned so use same intrinsics/distortion
        # https://support.stereolabs.com/hc/en-us/articles/4402102207383-What-are-the-advantages-of-using-the-Stereolabs-ZED-stereocameras-over-other-depth-cameras-eg-Intel-Realsense-available-in-the-market
        self.calibration["color"]["intrinsics"] = intrinsics
        self.calibration["color"]["distortion"] = distortion
        self.calibration["depth"]["intrinsics"] = intrinsics
        self.calibration["depth"]["distortion"] = distortion

        image = sl.Mat()
        depth = sl.Mat()
        while not self._halt:
            if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                # default to left image for now!
                self.camera.retrieve_image(image, sl.VIEW.LEFT)

                self.camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.last_obs = {"color": cv2.resize(image.get_data()[..., :3], (1280,720)).astype(np.uint8)}

        self.camera.close()

    def get_intrinsics(self, key, mode=None):
        if mode == "dict":
            return get_zed_intrinsics_param(self.calibration[key]["intrinsics"])
        else:
            return self.calibration[key]["intrinsics"]
        
    def get_distortion(self, key):
        return self.calibration[key]["distortion"]


class ZEDInterface:
    def __init__(self, device_id, resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.PERFORMANCE):
        self.camera = ZEDWorker(device_id=device_id, resolution=resolution, depth_mode=depth_mode)
        self.is_recording = False

        self.enable_color = True
        self.enable_depth = True

    def start(self):
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None or self.camera.last_obs == {}:
            return None
        else:
            self.last_obs = self.camera.last_obs
            return self.last_obs

    def close(self):
        self.camera.halt()

    def get_camera_intrinsics(self):
        return self.camera.get_intrinsics()

    def get_depth_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("depth", mode=mode)
        return intrinsics

    def get_color_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("color", mode=mode)
        return intrinsics

    def get_color_distortion(self, mode=None):
        distortion = self.camera.get_distortion("color")
        return distortion
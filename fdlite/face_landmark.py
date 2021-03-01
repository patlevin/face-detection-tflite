# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
import os
import numpy as np
import tensorflow as tf
from PIL.Image import Image
from typing import List, Optional, Tuple, Union
from fdlite.transform import SizeMode, bbox_to_roi, image_to_tensor, sigmoid
from fdlite.transform import project_landmarks
from fdlite.types import Detection, Landmark, Rect
from fdlite.face_detection import FaceIndex
"""Model for face landmark detection.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://mediapipe.page.link/facemesh-mc

Reference:

    Real-time Facial Surface Geometry from Monocular
    Video on Mobile GPUs, CVPR Workshop on Computer
    Vision for Augmented and Virtual Reality, Long Beach,
    CA, USA, 2019
"""

MODEL_NAME = 'face_landmark.tflite'
NUM_DIMS = 3                    # x, y, z
NUM_LANDMARKS = 468             # number of points in the face mesh
ROI_SCALE = (1.5, 1.5)          # Scaling of the face detection ROI
DETECTION_THRESHOLD = 0.5       # minimum score for detected faces


def face_detection_to_roi(
    face_detection: Detection,
    image_size: Tuple[int, int]
) -> Rect:
    """Return a normalized ROI from a list of face detection results.

    The result of this function is intended to serve as the input of
    calls to `FaceLandmark`:

    ```
        MODEL_PATH = '/var/mediapipe/models/'
        ...
        face_detect = FaceDetection(model_path=MODEL_PATH)
        face_landmarks = FaceLandmark(model_path=MODEL_PATH)
        image = Image.open('/home/user/pictures/photo.jpg')
        # detect faces
        detections = face_detect(image)
        for detection in detections:
            # find ROI from detection
            roi = face_detection_to_roi(detection)
            # extract face landmarks using ROI
            landmarks = face_landmarks(image, roi)
            ...
    ```

    Args:
        face_detection (Detection): Normalized face detection result from a
            call to `FaceDetection`.

        image_size (tuple): A tuple of `(image_width, image_height)` denoting
            the size of the input image the face detection results came from.

    Returns:
        (Rect) Normalized ROI for passing to `FaceLandmark`.
    """
    absolute_detection = face_detection.scaled(image_size)
    left_eye = absolute_detection[FaceIndex.LEFT_EYE]
    right_eye = absolute_detection[FaceIndex.RIGHT_EYE]
    return bbox_to_roi(
        face_detection.bbox,
        image_size,
        rotation_keypoints=[left_eye, right_eye],
        scale=ROI_SCALE,
        size_mode=SizeMode.SQUARE_LONG
    )


class FaceLandmark(object):
    """Face Landmark detection model as used by Google MediaPipe.

    This model detects facial landmarks from a face image.

    The model is callable and accepts a PIL image instance, image file name,
    and Numpy array of shape (height, width, channels) as input. There is no
    size restriction, but smaller images are processed faster.

    The output of the model is a list of 468 face landmarks in normalized
    coordinates (e.g. in the range [0, 1]).

    The preferred usage is to pass an ROI returned by a call to the
    `FaceDetection` model along with the image.
    """
    def __init__(
        self,
        model_path: Optional[str] = None
    ) -> None:
        if model_path is None:
            my_path = os.path.abspath(__file__)
            model_path = os.path.join(os.path.dirname(my_path), '../data')
        self.model_path = os.path.join(model_path, MODEL_NAME)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape']
        self.data_index = self.interpreter.get_output_details()[0]['index']
        self.face_index = self.interpreter.get_output_details()[1]['index']
        data_shape = self.interpreter.get_output_details()[0]['shape']
        num_exected_elements = NUM_DIMS * NUM_LANDMARKS
        if data_shape[-1] < num_exected_elements:
            raise ValueError(f'incompatible model: {data_shape} < '
                             f'{num_exected_elements}')
        self.interpreter.allocate_tensors()

    def __call__(
        self,
        image: Union[Image, np.ndarray, str],
        roi: Optional[Rect] = None
    ) -> List[Landmark]:
        """Run inference and return detections from a given image

        Args:
            image (Image|ndarray|str): Numpy array of shape
                `(height, width, 3)` or PIL Image instance or path to image.

            roi (Rect|None): Region within the image that contains a face.

        Returns:
            (list) List of face landmarks in nromalised coordinates relative to
            the input image, i.e. values ranging from [0, 1].
        """
        height, width = self.input_shape[1:3]
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=False,
            output_range=(0., 1.))
        input_data = image_data.tensor_data[np.newaxis]
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        raw_data = self.interpreter.get_tensor(self.data_index)
        raw_face = self.interpreter.get_tensor(self.face_index)
        # second tensor contains confidence score for a face detection
        face_flag = sigmoid(raw_face).flatten()[-1]
        # no data if no face was detected
        if face_flag <= DETECTION_THRESHOLD:
            return []
        # extract and normalise landmark data
        height, width = self.input_shape[1:3]
        return project_landmarks(raw_data,
                                 tensor_size=(width, height),
                                 image_size=image_data.original_size,
                                 padding=image_data.padding,
                                 roi=roi)

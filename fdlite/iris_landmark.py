# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import os
import tensorflow as tf
from PIL.Image import Image
from typing import List, Optional, Sequence, Tuple, Union
from fdlite import exif
from fdlite.render import Annotation, Color, Point, RectOrOval
from fdlite.render import landmarks_to_render_data
from fdlite.transform import bbox_from_landmarks, bbox_to_roi, image_to_tensor
from fdlite.transform import project_landmarks, SizeMode
from fdlite.types import Landmark, Rect
"""Iris landmark detection model.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://mediapipe.page.link/iris-mc

Reference:
    N/A
"""

MODEL_NAME = 'iris_landmark.tflite'
# ROI scale factor for 25% margin around eye
ROI_SCALE = (2.3, 2.3)
# Landmark index of the left eye start point
LEFT_EYE_START = 33
# Landmark index of the left eye end point
LEFT_EYE_END = 133
# Landmark index of the right eye start point
RIGHT_EYE_START = 362
# Landmark index of the right eye end point
RIGHT_EYE_END = 263
# Number of face landmarks (from face landmark results)
NUM_FACE_LANDMARKS = 468

# Landmark element count (x, y, z)
NUM_DIMS = 3
NUM_EYE_LANDMARKS = 71
NUM_IRIS_LANDMARKS = 5

# eye contour default visualisation settings
# (from iris_and_depth_renderer_cpu.pbtxt)
EYE_LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (11, 12), (12, 13), (13, 14), (0, 9), (8, 14)
]
MAX_EYE_LANDMARK = len(EYE_LANDMARK_CONNECTIONS)

# mapping from left eye contour index to face landmark index
LEFT_EYE_TO_FACE_LANDMARK_INDEX = [
    # eye lower contour
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # eye upper contour excluding corners
    246, 161, 160, 159, 158, 157, 173,
    # halo x2 lower contour
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    # halo x2 upper contour excluding corners
    247, 30, 29, 27, 28, 56, 190,
    # halo x3 lower contour
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    # halo x3 upper contour excluding corners
    113, 225, 224, 223, 222, 221, 189,
    # halo x4 upper contour (no upper due to mesh structure)
    # or eyebrow inner contour
    35, 124, 46, 53, 52, 65,
    # halo x5 lower contour
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    # halo x5 upper contour excluding corners or eyebrow outer contour
    156, 70, 63, 105, 66, 107, 55, 193,
]

# mapping from right eye contour index to face landmark index
RIGHT_EYE_TO_FACE_LANDMARK_INDEX = [
    # eye lower contour
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    # eye upper contour excluding corners
    466, 388, 387, 386, 385, 384, 398,
    # halo x2 lower contour
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    # halo x2 upper contour excluding corners
    467, 260, 259, 257, 258, 286, 414,
    # halo x3 lower contour
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    # halo x3 upper contour excluding corners
    342, 445, 444, 443, 442, 441, 413,
    # halo x4 upper contour (no upper due to mesh structure)
    # or eyebrow inner contour
    265, 353, 276, 283, 282, 295,
    # halo x5 lower contour
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    # halo x5 upper contour excluding corners or eyebrow outer contour
    383, 300, 293, 334, 296, 336, 285, 417,
]

# 35mm camera sensor diagonal (36mm * 24mm)
SENSOR_DIAGONAL_35MM = np.math.sqrt(36**2 + 24**2)
# average human iris size
IRIS_SIZE_IN_MM = 11.8


class IrisIndex(IntEnum):
    """Index into iris landmarks as returned by `IrisLandmark`
    """
    CENTER = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4


@dataclass
class IrisResults:
    """Iris detection results.

    contour data is 71 points defining the eye region

    iris data is 5 keypoints
    """
    contour: List[Landmark]
    iris: List[Landmark]

    @property
    def eyeball_contour(self) -> List[Landmark]:
        """Visible eyeball contour landmarks"""
        return self.contour[:MAX_EYE_LANDMARK]


def iris_roi_from_face_landmarks(
    face_landmarks: Sequence[Landmark],
    image_size: Tuple[int, int]
) -> Tuple[Rect, Rect]:
    """Extract iris landmark detection ROIs from face landmarks.

    Use this function to get the ROI for the left and right eye. The resulting
    bounding boxes are suitable for passing to `IrisDetection` as the ROI
    parameter. This is a pre-processing step originally found in the
    MediaPipe sub-graph "iris_landmark_landmarks_to_roi":

    ```
        def iris_detection(image, face_landmarks, iris_landmark_model):
            # extract left- and right eye ROI from face landmarks
            roi_left_eye, roi_right_eye = iris_roi_from_face_landmarks(
                face_landmarks)
            # use ROIs with iris detection
            iris_results_left = iris_landmark_model(image, roi_left_eye)
            iris_results_right = iris_landmark_model(image, roi_right_eye,
                                                    is_right_eye=True)
            return iris_result_left, iris_results_right
    ```

    Args:
        landmarks (list): Result of a `FaceLandmark` call containing face
            landmark detection results.

        image_size (tuple): Tuple of `(width, height)` representing the size
            of the image in pixels. The image must be the same as the one used
            for face lanmark detection.

    Return:
        (Rect, Rect) Tuple of ROIs containing the absolute pixel coordinates
        of the left and right eye regions. The ROIs can be passed to
        `IrisDetetion` together with the original image to detect iris
        landmarks.
    """
    left_eye_landmarks = (
        face_landmarks[LEFT_EYE_START],
        face_landmarks[LEFT_EYE_END])
    bbox = bbox_from_landmarks(left_eye_landmarks)
    rotation_keypoints = [(point.x, point.y) for point in left_eye_landmarks]
    w, h = image_size
    left_eye_roi = bbox_to_roi(
        bbox, (w, h), rotation_keypoints, ROI_SCALE, SizeMode.SQUARE_LONG)

    right_eye_landmarks = (
        face_landmarks[RIGHT_EYE_START],
        face_landmarks[RIGHT_EYE_END])
    bbox = bbox_from_landmarks(right_eye_landmarks)
    rotation_keypoints = [(point.x, point.y) for point in right_eye_landmarks]
    right_eye_roi = bbox_to_roi(
        bbox, (w, h), rotation_keypoints, ROI_SCALE, SizeMode.SQUARE_LONG)

    return left_eye_roi, right_eye_roi


def eye_landmarks_to_render_data(
    eye_contour: Sequence[Landmark],
    landmark_color: Color,
    connection_color: Color,
    thickness: float = 2.0,
    output: Optional[List[Annotation]] = None
) -> List[Annotation]:
    """Convert eye contour to render data.

    This post-processing function can be used to generate a list of rendering
    instructions from iris landmark detection results.

    Args:
        eye_contour (list): List of `Landmark` detection results returned
            by `IrisLandmark`.

        landmark_color (Color): Color of the individual landmark points.

        connection_color (Color): Color of the landmark connections that
            will be rendered as lines.

        thickness (float): Width of the lines and landmark point size in
            viewport units (e.g. pixels).

        output (list): Optional list of render annotations to add the items
            to. If not provided, a new list will be created.
            Use this to add multiple landmark detections into a single render
            annotation list.

    Returns:
        (list) List of render annotations that should be rendered.
        All positions are normalized, e.g. with a value range of [0, 1].
    """
    render_data = landmarks_to_render_data(
        eye_contour[0:MAX_EYE_LANDMARK], EYE_LANDMARK_CONNECTIONS,
        landmark_color, connection_color, thickness,
        normalized_positions=True, output=output)
    return render_data


def iris_landmarks_to_render_data(
    iris_landmarks: Sequence[Landmark],
    landmark_color: Optional[Color] = None,
    oval_color: Optional[Color] = None,
    thickness: float = 1.0,
    image_size: Tuple[int, int] = (-1, -1),
    output: Optional[List[Annotation]] = None
) -> List[Annotation]:
    """Convert iris landmarks to render data.

    This post-processing function can be used to generate a list of rendering
    instructions from iris landmark detection results.

    Args:
        iris_landmarks (list): List of `Landmark` detection results returned
            by `IrisLandmark`.

        landmark_color (Color|None): Color of the individual landmark points.

        oval_color (Color|None): Color of the iris oval.

        thickness (float): Width of the iris oval and landmark point size
            in viewport units (e.g. pixels).

        image_size (tuple): Image size as a tuple of `(width, height)`.

        output (list|None): Optional list of render annotations to add the
            items to. If not provided, a new list will be created.
            Use this to add multiple landmark detections into a single render
            annotation list.

    Returns:
        (list) Render data bundle containing points for landmarks and
        lines for landmark connections. All positions are normalised with
        respect to the output viewport (e.g. value range is [0, 1])
    """
    annotations = []
    if oval_color is not None:
        iris_radius = _get_iris_diameter(iris_landmarks, image_size) / 2
        width, height = image_size
        if width < 2 or height < 2:
            raise ValueError('invalid image_size')
        radius_h = iris_radius / width
        radius_v = iris_radius / height
        iris_center = iris_landmarks[IrisIndex.CENTER]
        oval = RectOrOval(iris_center.x - radius_h, iris_center.y - radius_v,
                          iris_center.x + radius_h, iris_center.y + radius_v,
                          oval=True)
        annotations.append(Annotation([oval], normalized_positions=True,
                                      thickness=thickness, color=oval_color))
    if landmark_color is not None:
        points = [Point(pt.x, pt.y) for pt in iris_landmarks]
        annotations.append(Annotation(points, normalized_positions=True,
                                      thickness=thickness,
                                      color=landmark_color))
    if output is None:
        output = annotations
    else:
        output += annotations
    return output


def update_face_landmarks_with_iris_results(
    face_landmarks: Sequence[Landmark],
    iris_data_left: IrisResults,
    iris_data_right: IrisResults
) -> List[Landmark]:
    """Update face landmarks with iris detection results.

    Landmarks will be updated with refined results from iris tracking.

    Args:
        face_landmarks (list): Face landmark detection results with
            coarse eye landmarks

        iris_data_left (IrisResults): Iris landmark results for the left eye

        iris_data_right (IrisResults): Iris landmark results for the right eye

    Returns:
        (list) List of face landmarks with refined eye contours and iris
        landmarks
    """
    if len(face_landmarks) != NUM_FACE_LANDMARKS:
        raise ValueError('unexpected number of items in face_landmarks')
    # copy landmarks
    refined_landmarks = [Landmark(pt.x, pt.y, pt.z) for pt in face_landmarks]
    # merge left eye contours
    for n, point in enumerate(iris_data_left.contour):
        index = LEFT_EYE_TO_FACE_LANDMARK_INDEX[n]
        refined_landmarks[index] = Landmark(point.x, point.y, point.z)
    # merge right eye contours
    for n, point in enumerate(iris_data_right.contour):
        index = RIGHT_EYE_TO_FACE_LANDMARK_INDEX[n]
        refined_landmarks[index] = Landmark(point.x, point.y, point.z)
    return refined_landmarks


def iris_depth_in_mm_from_landmarks(
    image_or_focal_length: Union[Image, Tuple[int, int, int, int]],
    iris_data_left: IrisResults,
    iris_data_right: IrisResults
) -> Tuple[float, float]:
    """Calculate the distances to the left- and right eye from image meta data
    and iris landmarks.

    The calculation requires EXIF meta data to be present if an image is
    provided. Alternatively, focal length and image size in pixels can be
    provided directly.

    Args:
        image_or_focal_length (Image|tuple): Either a PIL image instance with
            EXIF meta data or a tuple of
            `(focal_length_35mm, focal_length_mm, image_width, image_height)`

        iris_data_left (IrisResults): Detection results from `IrisLandmark`
            for the left eye.

        iris_data_right (IrisResults): Detection results from `IrisLandmark`
            for the right eye.

    Raises:
        ValueError: `image_or_sensor_data` is a PIL image without the required
            EXIF meta data or the provided tuple is malformed (e.g. mismatch
            in expected number of elements).

    Returns:
        (tuple) Tuple of `(left_eye_distance_in_mm, right_eye_distance_in_mm)`
    """
    if isinstance(image_or_focal_length, Image):
        from_exif = exif.get_focal_length(image_or_focal_length)
        if from_exif is None:
            raise ValueError('no focal length in EXIF data or unknown camera')
        focal_length = from_exif
    else:
        focal_length = image_or_focal_length
    if len(focal_length) != 4:
        raise ValueError('image_or_focal_length tuple element count mismatch')
    # calculate focal length in pixels
    focal_len_35mm, focal_len_mm, width_px, height_px = focal_length
    sensor_diagonal_mm = SENSOR_DIAGONAL_35MM / focal_len_35mm * focal_len_mm
    w, h = width_px, height_px
    pixel_size = (width_px, height_px)
    # treat the shorter dimension as width
    if height_px > width_px:
        w, h = h, w
    sqr_inv_aspect = (h / w) ** 2
    sensor_width = np.math.sqrt((sensor_diagonal_mm**2) / (1 + sqr_inv_aspect))
    focal_len_px = w * focal_len_mm / sensor_width
    left_landmarks, right_landmarks = iris_data_left.iris, iris_data_right.iris
    left_iris_size = _get_iris_diameter(left_landmarks, pixel_size)
    right_iris_size = _get_iris_diameter(right_landmarks, pixel_size)
    left_depth_mm = _get_iris_depth(left_landmarks, focal_len_px,
                                    left_iris_size, pixel_size)
    right_depth_mm = _get_iris_depth(right_landmarks, focal_len_px,
                                     right_iris_size, pixel_size)
    return left_depth_mm, right_depth_mm


class IrisLandmark(object):
    """Model for iris landmark detection from the image of an eye.

    The model expects the image of an eye as input, complete with brows and
    a 25% margin around the eye.

    The outputs of the model are 71 normalized eye contour landmarks and a
    separate list of 5 normalized iris landmarks.

    The model is callable and accepts a PIL image instance, image file name,
    and Numpy array of shape (height, width, channels) as input. There is no
    size restriction, but smaller images are processed faster.

    The provided image either matches the model's input spec or an ROI is
    provided, which denotes the eye location within the image.
    The ROI can be obtained from calling the `FaceDetection` model with the
    image and converting the iris ROI from the result:

    ```
        MODEL_PATH = '/var/mediapipe/models'

        img = PIL.Image.open('/home/usr/pictures/group.jpg', mode='RGB')
        # 1) load models
        detect_face = FaceDetection(model_path=MODEL_PATH)
        detect_face_landmarks = FaceLandmark(model_path=MODEL_PATH)
        detect_iris_landmarks = IrisLandmark(model_path=MODEL_PATH)
        # 2) run face detection
        face_detections = detect_face(img)
        # 3) detect face landmarks
        for detection in face_detections:
            face_roi = face_detection_to_roi(detection, img.size)
            face_landmarks = detect_face_landmarks(img, face_roi)
            # 4) run iris detection
            iris_roi = iris_roi_from_landmarks(face_landmarks, img.size)
            left_eye_detection = detect_iris_landmarks(img, iris_roi[0])
            right_eye_detection = detect_iris_landmarks(
                img, iris_roi[1], is_right_eye=True)
            ...
    ```
    """
    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            my_path = os.path.abspath(__file__)
            model_path = os.path.join(os.path.dirname(my_path), 'data')
        self.model_path = os.path.join(model_path, MODEL_NAME)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape']
        self.eye_index = self.interpreter.get_output_details()[0]['index']
        self.iris_index = self.interpreter.get_output_details()[1]['index']

        eye_shape = self.interpreter.get_output_details()[0]['shape']
        if eye_shape[-1] != NUM_DIMS * NUM_EYE_LANDMARKS:
            raise ValueError('unexpected number of eye landmarks: '
                             f'{eye_shape[-1]}')
        iris_shape = self.interpreter.get_output_details()[1]['shape']
        if iris_shape[-1] != NUM_DIMS * NUM_IRIS_LANDMARKS:
            raise ValueError('unexpected number of iris landmarks: '
                             f'{eye_shape[-1]}')
        self.interpreter.allocate_tensors()

    def __call__(
        self,
        image: Union[Image, np.ndarray, str],
        roi: Optional[Rect] = None,
        is_right_eye: bool = False
    ) -> IrisResults:
        height, width = self.input_shape[1:3]
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=True,     # equivalent to scale_mode=FIT
            output_range=(0, 1),        # see iris_landmark_cpu.pbtxt
            flip_horizontal=is_right_eye
        )
        input_data = image_data.tensor_data[np.newaxis]
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        raw_eye_landmarks = self.interpreter.get_tensor(self.eye_index)
        raw_iris_landmarks = self.interpreter.get_tensor(self.iris_index)
        height, width = self.input_shape[1:3]
        eye_contour = project_landmarks(
            raw_eye_landmarks,
            tensor_size=(width, height),
            image_size=image_data.original_size,
            padding=image_data.padding,
            roi=roi,
            flip_horizontal=is_right_eye)
        iris_landmarks = project_landmarks(
            raw_iris_landmarks,
            tensor_size=(width, height),
            image_size=image_data.original_size,
            padding=image_data.padding,
            roi=roi,
            flip_horizontal=is_right_eye)
        return IrisResults(eye_contour, iris_landmarks)


def _get_iris_diameter(
    iris_landmarks: Sequence[Landmark], image_size: Tuple[int, int]
) -> float:
    """Calculate the iris diameter in pixels"""
    width, height = image_size

    def get_landmark_depth(a: Landmark, b: Landmark) -> float:
        x0, y0, x1, y1 = a.x * width, a.y * height, b.x * width, b.y * height
        return np.math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    iris_size_horiz = get_landmark_depth(iris_landmarks[IrisIndex.LEFT],
                                         iris_landmarks[IrisIndex.RIGHT])
    iris_size_vert = get_landmark_depth(iris_landmarks[IrisIndex.TOP],
                                        iris_landmarks[IrisIndex.BOTTOM])
    return (iris_size_vert + iris_size_horiz) / 2


def _get_iris_depth(
    iris_landmarks: Sequence[Landmark],
    focal_length_mm: float,
    iris_size_px: float,
    image_size: Tuple[int, int]
) -> float:
    """Calculate iris depth in mm from landmarks and lens focal length in mm"""
    width, height = image_size
    center = iris_landmarks[IrisIndex.CENTER]
    x0, y0 = width / 2, height / 2
    x1, y1 = center.x * width, center.y * height
    y = np.math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    x = np.math.sqrt(focal_length_mm ** 2 + y ** 2)
    return IRIS_SIZE_IN_MM * x / iris_size_px

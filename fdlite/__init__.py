# -*- coding: utf-8 -*-
from .face_detection import FaceDetection, FaceDetectionModel     # noqa:F401
from .face_detection import FaceIndex                             # noqa:F401
from .face_landmark import FaceLandmark, face_detection_to_roi    # noqa:F401
from .face_landmark import face_landmarks_to_render_data          # noqa:F401
from .iris_landmark import IrisIndex, IrisLandmark, IrisResults   # noqa:F401
from .iris_landmark import eye_landmarks_to_render_data           # noqa:F401
from .iris_landmark import iris_depth_in_mm_from_landmarks        # noqa:F401
from .iris_landmark import iris_landmarks_to_render_data          # noqa:F401
from .iris_landmark import iris_roi_from_face_landmarks           # noqa:F401


class InvalidEnumError(Exception):
    """Raised when a function was called with an invalid Enum value"""


class ModelDataError(Exception):
    """Raised when a model returns data that is incompatible"""


class CoordinateRangeError(Exception):
    """Raised when coordinates are expected to be in a different range"""


class ArgumentError(Exception):
    """Raised when an argument is of the wrong type or malformed"""


class MissingExifDataError(Exception):
    """Raised if required EXIF data is missing from an image"""


__version__ = '0.4.0'

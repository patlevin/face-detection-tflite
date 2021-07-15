# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
from PIL import ImageDraw
from PIL.Image import Image as PILImage
from fdlite import CoordinateRangeError
from fdlite.types import Detection, Landmark
"""Types and functions related to rendering detection results"""


@dataclass
class Color:
    """Color for rendering annotations"""
    r: int = 0
    g: int = 0
    b: int = 0
    a: Optional[int] = None

    @property
    def as_tuple(
        self
    ) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Return color components as tuple"""
        r, g, b, a = self.r, self.g, self.b, self.a
        return (r, g, b) if a is None else (r, g, b, a)


class Colors:
    """Predefined common color values for use with annotations rendering"""
    BLACK = Color()
    RED = Color(r=255)
    GREEN = Color(g=255)
    BLUE = Color(b=255)
    PINK = Color(r=255, b=255)
    WHITE = Color(r=255, g=255, b=255)


@dataclass
class Point:
    """A point to be rendered"""
    x: float
    y: float

    @property
    def as_tuple(self) -> Tuple[float, float]:
        """Values as a tuple of (x, y)"""
        return self.x, self.y

    def scaled(self, factor: Tuple[float, float]) -> 'Point':
        """Return a point with an absolute position"""
        sx, sy = factor
        return Point(self.x * sx, self.y * sy)


@dataclass
class RectOrOval:
    """A rectangle or oval to be rendered"""
    left: float
    top: float
    right: float
    bottom: float
    oval: bool = False

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Values as a tuple of (left, top, right, bottom)"""
        return self.left, self.top, self.right, self.bottom

    def scaled(self, factor: Tuple[float, float]) -> 'RectOrOval':
        """Return a rect or oval with absolute positions"""
        sx, sy = factor
        return RectOrOval(self.left * sx,  self.top * sy,
                          self.right * sx, self.bottom * sy, self.oval)


@dataclass
class FilledRectOrOval:
    """A filled rectangle or oval to be rendered"""
    rect: RectOrOval
    fill: Color

    def scaled(self, factor: Tuple[float, float]) -> 'FilledRectOrOval':
        """Return a filled rect or oval with absolute positions"""
        return FilledRectOrOval(self.rect.scaled(factor), self.fill)


@dataclass
class Line:
    """A solid or dashed line to be rendered"""
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    dashed: bool = False

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Values as a tuple of (x_start, y_start, x_end, y_end)"""
        return self.x_start, self.y_start, self.x_end, self.y_end

    def scaled(self, factor: Tuple[float, float]) -> 'Line':
        """Return line with absolute positions"""
        sx, sy = factor
        return Line(self.x_start * sx, self.y_start * sy,
                    self.x_end * sx, self.y_end * sy, self.dashed)


@dataclass
class Annotation:
    """Graphical annotation to be rendered

    Massively cut-down version of the MediaPipe type.
    Annotation data is bundled for higher data efficiency.
    Normalisation flag, thickness, and color apply to all
    items in the data list to reduce redundancy.

    The corresponding converter functions will automatically
    bundle data by type and format.
    """
    data: Sequence[Union[Point, RectOrOval, FilledRectOrOval, Line]]
    normalized_positions: bool
    thickness: float
    color: Color

    def scaled(self, factor: Tuple[float, float]) -> 'Annotation':
        """Return an annotation with all positions scaled yb a given factor

        Args:
            factor (tuple): Scaling factor as a tuple of `(width, height)`.

        Raises:
            CoordinateRangeError: position data is not normalized

        Returns:
            (Annotation) Annotation with all elements scaled to the given
            size.
        """
        if not self.normalized_positions:
            raise CoordinateRangeError('position data must be normalized')
        scaled_data = [item.scaled(factor) for item in self.data]
        return Annotation(scaled_data, normalized_positions=False,
                          thickness=self.thickness, color=self.color)


def detections_to_render_data(
    detections: Sequence[Detection],
    bounds_color: Optional[Color] = None,
    keypoint_color: Optional[Color] = None,
    line_width: int = 1,
    point_width: int = 3,
    normalized_positions: bool = True,
    output: Optional[List[Annotation]] = None
) -> List[Annotation]:
    """Convert detections to render data.

    This is an implementation of the MediaPipe DetectionToRenderDataCalculator
    node with keypoints added.

    Args:
        detections (list): List of detects, which will be converted to
            individual points and a bounding box rectangle for rendering.

        bounds_color (RenderColor|None): Color of the bounding box; if `None`
            the bounds won't be rendered.

        keypoint_color (RenderColor|None): Color of the keypoints that will
            will be rendered as points; set to `None` to disable keypoint
            rendering.

        line_width (int): Thickness of the lines in viewport units
            (e.g. pixels).

        point_width (int): Size of the keypoints in viewport units
            (e.g. pixels).

        normalized_positions (bool): Flag indicating whether the detections
            contain normalised data (e.g. range [0,1]).

        output (RenderData): Optional render data instance to add the items
            to. If not provided, a new instance of `RenderData` will be
            created.
            Use this to add multiple landmark detections into a single render
            data bundle.

    Returns:
        (list) List of annotations for rendering landmarks.
    """

    def to_rect(detection: Detection) -> RectOrOval:
        bbox = detection.bbox
        return RectOrOval(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)

    annotations = []
    if bounds_color is not None and line_width > 0:
        bounds = Annotation([to_rect(detection) for detection in detections],
                            normalized_positions, thickness=line_width,
                            color=bounds_color)
        annotations.append(bounds)
    if keypoint_color is not None and point_width > 0:
        points = Annotation([Point(x, y)
                            for detection in detections
                            for (x, y) in detection],
                            normalized_positions, thickness=point_width,
                            color=keypoint_color)
        annotations.append(points)
    if output is not None:
        output += annotations
    else:
        output = annotations
    return output


def landmarks_to_render_data(
    landmarks: Sequence[Landmark],
    landmark_connections: Sequence[Tuple[int, int]],
    landmark_color: Color = Colors.RED,
    connection_color: Color = Colors.RED,
    thickness: float = 1.,
    normalized_positions: bool = True,
    output: Optional[List[Annotation]] = None
) -> List[Annotation]:
    """Convert detected landmarks to render data.

    This is an implementation of the MediaPipe LandmarksToRenderDataCalculator
    node.

    Args:
        landmarks (list): List of detected landmarks, which will be converted
            to individual points for rendering.

        landmark_connections (list): List of tuples in the form of
            `(offset_from, offset_to)` that represent connections between
            landmarks. The offsets are zero-based indexes into `landmarks`
            and each tuple will be converted to a line for rendering.

        landmark_color (RenderColor): Color of the individual landmark points.

        connection_color (RenderColor): Color of the landmark connections that
            will be rendered as lines.

        thickness (float): Thickness of the lines and landmark point size in
            viewport units (e.g. pixels).

        normalized_positions (bool): Flag indicating whether the landmarks
            contain normalised data (e.g. range [0,1]).

        output (RenderData): Optional render data instance to add the items
            to. If not provided, a new instance of `RenderData` will be
            created.
            Use this to add multiple landmark detections into a single render
            data bundle.

    Returns:
        (list) List of annotations for rendering landmarks.
    """
    lm = landmarks
    lines = [Line(lm[start].x, lm[start].y, lm[end].x, lm[end].y)
             for start, end in landmark_connections]
    points = [Point(landmark.x, landmark.y) for landmark in landmarks]
    la = Annotation(lines, normalized_positions, thickness, connection_color)
    pa = Annotation(points, normalized_positions, thickness, landmark_color)
    if output is not None:
        output += [la, pa]
    else:
        output = [la, pa]
    return output


def render_to_image(
    annotations: Sequence[Annotation],
    image: PILImage,
    blend: bool = False
) -> PILImage:
    """Render annotations to an image.

    Args:
        annotations (list): List of annotations.

        image (Image): PIL Image instance to render to.

        blend (bool): If `True`, allows for alpha-blending annotations on
            top of the image.

    Returns:
        (Image) Returns the modified image.
    """
    draw = ImageDraw.Draw(image, mode='RGBA' if blend else 'RGB')
    for annotation in annotations:
        if annotation.normalized_positions:
            scaled = annotation.scaled(image.size)
        else:
            scaled = annotation
        if not len(scaled.data):
            continue
        thickness = int(scaled.thickness)
        color = scaled.color
        for item in scaled.data:
            if isinstance(item, Point):
                w = max(thickness // 2, 1)
                rc = [item.x - w, item.y - w, item.x + w, item.y + w]
                draw.rectangle(rc, fill=color.as_tuple, outline=color.as_tuple)
            elif isinstance(item, Line):
                coords = [item.x_start, item.y_start, item.x_end, item.y_end]
                draw.line(coords, fill=color.as_tuple, width=thickness)
            elif isinstance(item, RectOrOval):
                rc = [item.left, item.top, item.right, item.bottom]
                if item.oval:
                    draw.ellipse(rc, outline=color.as_tuple, width=thickness)
                else:
                    draw.rectangle(rc, outline=color.as_tuple, width=thickness)
            elif isinstance(item, FilledRectOrOval):
                rgb = color.as_tuple
                rect, fill = item.rect, item.fill.as_tuple
                rc = [rect.left, rect.top, rect.right, rect.bottom]
                if rect.oval:
                    draw.ellipse(rc, fill=fill, outline=rgb, width=thickness)
                else:
                    draw.rectangle(rc, fill=fill, outline=rgb, width=thickness)
            else:
                # don't know how to render this
                pass
    return image

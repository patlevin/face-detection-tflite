# Face Detection Package Tutorial

This package is a port of some [**Google® MediaPipe**](https://google.github.io/mediapipe/)
graphs to simple Python functions. The goal is to provide access to some of
the pretrained models used in MediaPipe without the complexities of the
graph concepts used.

## A Word on Coordinates

The library uses custom types which by default only hold normalized
coordinates. This means the values for x and y range from 0..1 and are
relative to the input image. The reasoning behind this choice is
flexibility. Normalized coordinates don't need to change if you scale
the image (e.g.for displaying purposes).

Most types contain a `scale()`-method that can be used to scale the
coordinates to the requested (image-)size.

## Detecting Faces

You can use the `face_detection` module to find faces within an image.
The `FaceDetection` model will return a list of `Detection`s for each face
found. These detections are *normalized*, meaning the coordinates range from
0..1 and are relative to the input image.

Each model class is callable, meaning once instanciated you can call them
just like a function. The example below shows how to detect faces and
display results using `Pillow`:

```python
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image
from PIL import Image

# load the model; select "back camera"-model for groups and smaller faces
detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
# open an image
img = Image.open('group.jpg')
# detect faces
detections = detect_faces(img)
if len(detections):
    # convert results to render data; show bounding boxes only
    render_data = detections_to_render_data(
        detections, bounds_color=Colors.GREEN, line_width=4)
    # render to image and display results
    render_to_image(render_data, img).show()
else:
    print('no faces found :(')
```

The result could look something like this:

![Group photo with face detections](group_photo.jpg)

*image source: [pexels.com](https://www.pexels.com/photo/group-of-people-watching-on-laptop-1595385/)*

## Detecting Face Landmarks

The face detection model only produces bounding boxes and crude keypoints.
A detailed 3D face mesh with over 480 landmarks can be obtained by using the
`FaceLandmark` model found in the `face-landmark` module. The recommended
use of this model is to calculate a *region of interest* (ROI) from the
output of the `FaceDetection` model and use it as an input:

> **FaceDetection(image) ⇒ ROI from detection ⇒ FaceLandmarks(image, ROI)**

Face landmark detection is fairly simple:

```python
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from PIL import Image

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()

# open image; by default, the "front camera"-model is used, which is smaller
# and ideal for selfies, and close-up portraits
img = Image.open('portrait.jpg')
# detect face
face_detections = detect_faces(img)
if len(face_detections):
    # get ROI for the first face found
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    # detect face landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    # convert detections to render data
    render_data = landmarks_to_render_data(
        face_landmarks, [], landmark_color=Colors.PINK, thickness=3)
    # render and display landmarks (points only)
    render_to_image(render_data, img).show()
else:
    print('no face detected :(') 
```

The result could look something like this:

![Portrait with face landmarks](portrait_fl.jpg)

*Photo by Andrea Piacquadio from [Pexels](https://www.pexels.com/photo/brown-freckles-on-face-3763152/)*

## Iris Detection

The final model in this package is iris detection. Just like face landmark
detection, this model is best used with an **ROI** obtained from face
landmarks. The processing chain from image to iris landmarks looks like this:

> **FaceDetection ⇒ ROI from detection ⇒ FaceLandmark ⇒ ROI from landmarks ⇒ IrisDetection**

Iris detection results consist of two sets of landmarks: one contains the
basic keypoints (eye boundary points and pupil center), while the other is a
refined version of the landmarks returned by `FaceLandmark`.

```python
# for face landmark detection
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
# for iris landmark detection
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
# for eye landmark rendering 
from fdlite import eye_landmarks_to_render_data
# for rendering to image
from fdlite.render import Colors, render_to_image
# for finding just the eye region in the image
from fdlite.transform import bbox_from_landmarks
from PIL import Image

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()
detect_iris = IrisLandmark()

# open image
img = Image.open('portrait.jpg')
# detect face
face_detections = detect_faces(img)
if len(face_detections):
    # get ROI for the first face found
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    # detect face landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    # get ROI for both eyes
    eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
    left_eye_roi, right_eye_roi = eye_roi
    # detect iris landmarks for both eyes
    left_eye_results = detect_iris(img, left_eye_roi)
    right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
    # convert landmarks to render data
    render_data = eye_landmarks_to_render_data(left_eye_results.contour,
                                               landmark_color=Colors.PINK,
                                               connection_color=Colors.GREEN)
    # add landmarks of the right eye
    _ = eye_landmarks_to_render_data(right_eye_results.contour,
                                     landmark_color=Colors.PINK,
                                     connection_color=Colors.GREEN,
                                     output=render_data)
    # render to image
    render_to_image(render_data, img)
    # get the bounds of just the eyes
    contours = left_eye_results.contour + right_eye_results.contour
    eye_box = bbox_from_landmarks(contours).absolute(img.size)
    # calculate a scaled-up version of the eye region
    new_size = (int(eye_box.width * 2), int(eye_box.height * 2))
    # isolate the eyes, scale them up, and display the result
    img.crop(eye_box.as_tuple).resize(new_size).show()
else:
    print('no face detected :(')
```

The result will look something like this:

![Eyes with detected contours marked](eyes.jpg)

## Fun with Iris Detection

The package includes a rudimentary exmaple of iris recoloring using iris
detection results. The function `recolor_iris` from the `examples` module
accepts iris detections results and a tuple of `(red, green, blue)` for
simple iris recoloring:

```python
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite.examples import recolor_iris
from PIL import Image

EXCITING_NEW_EYE_COLOR = (161, 52, 216)

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()
detect_iris = IrisLandmark()

# open image
img = Image.open('portrait.jpg')
# detect face
face_detections = detect_faces(img)
if len(face_detections):
    # get ROI for the first face found
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    # detect face landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    # get ROI for both eyes
    eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
    left_eye_roi, right_eye_roi = eye_roi
    # detect iris landmarks for both eyes
    left_eye_results = detect_iris(img, left_eye_roi)
    right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
    # change the iris color
    recolor_iris(img, left_eye_results, iris_color=EXCITING_NEW_EYE_COLOR)
    recolor_iris(img, right_eye_results, iris_color=EXCITING_NEW_EYE_COLOR)
    img.show()
else:
    print('no face detected :(')
```

The result will look similar to this:

![Recolored iris (cropped; with vignette effect)](recolored.jpg)

(cropped with vignette applied to distract from the inaccurracies present in the output)

## Estimating Eye Distance from Camera

The package also contains a function to estimate the distance of the eyes to
the camera. This comes with several caveats, though. Firstly, the image must
contain additional information about the lens and the sensor size. This data
(EXIF) is usually stored by smartphones and professional photo cameras but
might be missing from downloaded pictures. In this case, the information
can be provided manually as well.

The package contains a library of camera models to look up sensor size
information if it is missing from the EXIF data (focal length is still
required to be present).

To get a distance estimation, first detect the iris data and then pass it to
`iris_depth_in_mm_from_landmarks` along with the image:

```python
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite import iris_depth_in_mm_from_landmarks
from PIL import Image

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()
detect_iris = IrisLandmark()

# open image
img = Image.open('portrait_exif.jpg')
# detect face
face_detections = detect_faces(img)
if len(face_detections):
    # get ROI for the first face found
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    # detect face landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    # get ROI for both eyes
    eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
    left_eye_roi, right_eye_roi = eye_roi
    # detect iris landmarks for both eyes
    left_eye_results = detect_iris(img, left_eye_roi)
    right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
    # change the iris color
    dist_left_mm, dist_right_mm = iris_depth_in_mm_from_landmarks(
        img, left_eye_results, right_eye_results)
    print(f'Distance from camera appr. {dist_left_mm//10} cm to '
          f'{dist_right_mm//10} cm')
else:
    print('no face detected :(')
```

Given the image below
![Portrait with EXIF data](portrait_exif.jpg)

(source: [pixnio.com](https://pixnio.com/people/female-women/portrait-people-girl-woman-fashion-face-shadow-person))

will result in the following output:

> Distance from camera appr. 95.0 cm to 99.0 cm

Please note that this feature works best with smartphone photos and can
be very inaccurate with pictures taken using professional equipment.
This is due to the various possibilities that these cameras provide (lenses,
optical and digital zoom, etc.) Using values obtained from tools like
`Exiftool` will often yield vastly different results.

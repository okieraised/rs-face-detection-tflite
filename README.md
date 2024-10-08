# rs-face-detection-tflite

This package, inspired by the patlevin's [**face-detection-tflite**](https://github.com/patlevin/face-detection-tflite),
also tries to implement parts of Google®'s [**MediaPipe**](https://mediapipe.dev/#!) models in 
Rust using OpenCV and rust-ndarray.

## Models and Examples

The package provides the following models:

* Face Detection
    ![Example](https://github.com/okieraised/rs-face-detection-tflite/blob/main/assets/man_bbox.png)

* Face Landmark Detection
    ![Example](https://github.com/okieraised/rs-face-detection-tflite/blob/main/assets/man_landmark.png)

* Iris Landmark Detection
    ![Example](https://github.com/okieraised/rs-face-detection-tflite/blob/main/assets/man_iris.png)

To run the detection:
```rust
let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();

// test data
let im_bytes: &[u8] = include_bytes!("../test_data/man.jpg");
let image = convert_image_to_mat(im_bytes).unwrap();
let img_shape = image.size().unwrap();

// Face detection
let faces = face_detection.infer(&image, None).unwrap();
let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();

// Face landmark
let face_landmark = FaceLandmark::new(None).unwrap();
let lmks = face_landmark.infer(&image, Some(face_roi)).unwrap();

// Face irises
let (left_eye_roi, right_eye_roi) = iris_roi_from_face_landmarks(lmks.clone(), (img_shape.width, img_shape.height)).unwrap();
let iris_landmark = IrisLandmark::new(None).unwrap();

let right_iris_lmk = iris_landmark.infer(&image, Some(right_eye_roi), Some(true)).unwrap();
let left_iris_lmk = iris_landmark.infer(&image, Some(left_eye_roi), Some(false)).unwrap();
```

## Installation
* OpenCV, as well as opencv-rust library is required. For installation guide, please take a look at [**opencv-rust**](https://github.com/twistedfall/opencv-rust)

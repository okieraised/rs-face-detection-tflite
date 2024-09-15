# rs-face-detection-tflite

This package, inspired by the patlevin's [**face-detection-tflite**](https://github.com/patlevin/face-detection-tflite),
also tries to implement parts of Google®'s [**MediaPipe**](https://mediapipe.dev/#!) models in 
Rust using OpenCV and rust-ndarray.

## Models and Examples

The package provides the following models:

* Face Detection
    [Example](assets/man_bbox.png)

* Face Landmark Detection
    [Example](assets/man_landmark.png)

* Iris Landmark Detection

To run the detection:
```rust
let face_detection = FaceDetection::new(
    FaceDetectionModel::BackCamera,
    Some("/Users/tripham/Desktop/rs-face-detection-tflite/src/models".to_string()),
)
.unwrap();

let im_bytes: &[u8] = include_bytes!("/Users/tripham/Downloads/man.jpg");
let image = convert_image_to_mat(im_bytes).unwrap();
let img_shape = image.size().unwrap();

let faces = face_detection.infer(&image, None).unwrap();

let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();

let render_data = detections_to_render_data(
    faces,
    Some(Colors::GREEN),
    None,
    4,
    2,
    true,
    None,
);

let img = ImageReader::open("/Users/tripham/Downloads/man.jpg")
            .unwrap()
            .decode()
            .unwrap();

let res = render_to_image(&render_data, &img, None);
res.save("/Users/tripham/Downloads/test_man.png").unwrap();

```

## Installation
* OpenCV, as well as opencv-rust library is required. For installation guide, please take a look at [**opencv-rust**](https://github.com/twistedfall/opencv-rust)

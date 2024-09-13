use crate::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel, FaceIndex};
use crate::face_detection_lite::render::{landmarks_to_render_data, Annotation, Color};
use crate::face_detection_lite::transform::{bbox_to_roi, image_to_tensor, project_landmarks, sigmoid, SizeMode};
use crate::face_detection_lite::types::{Detection, Landmark, Rect};
use anyhow::Error;
use ndarray::{Array4, Axis};
use opencv::core::Mat;
use std::path::PathBuf;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};

/// Model for face landmark detection.
///
/// Ported from Google® MediaPipe (https://google.github.io/mediapipe/).
///
/// Model card:
///
///     https://mediapipe.page.link/facemesh-mc
///
/// Reference:
///
///     Real-time Facial Surface Geometry from Monocular
///     Video on Mobile GPUs, CVPR Workshop on Computer
///     Vision for Augmented and Virtual Reality, Long Beach,
///     CA, USA, 2019


const MODEL_NAME: &str = "face_landmark.tflite";
const NUM_DIMS: i32 = 3;
const NUM_LANDMARKS: i32 = 468;
const ROI_SCALE: (f64, f64) = (1.5, 1.5);
const DETECTION_THRESHOLD: f32 = 0.5;

/// face landmark connections
/// (from face_landmarks_to_render_data_calculator.cc)
const FACE_LANDMARK_CONNECTIONS: [(i32, i32); 124] = [
    // Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    // Left eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    // Left eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    // Right eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    // Right eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    // Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
];

const MAX_FACE_LANDMARK: usize = FACE_LANDMARK_CONNECTIONS.len();

/// Return a normalized ROI from a list of face detection results.
///
/// The result of this function is intended to serve as the input of calls to `FaceLandmark`:
/// * Args:
///     - face_detection (`Detection`): Normalized face detection result from a  call to `FaceDetection`.
///     - image_size (`(i32, i32)`): A tuple of `(image_width, image_height)` denoting
///       the size of the input image the face detection results came from.
///
/// * Returns:
///     - `Rect`: Normalized ROI for passing to `FaceLandmark`.
pub fn face_detection_to_roi(face_detection: Detection, image_size: (i32, i32)) -> Result<Rect, Error> {
    let absolute_detection: Detection = face_detection.scaled_by_image_size((image_size.0, image_size.1));
    let left_eye_f32 = absolute_detection.keypoint(FaceIndex::LeftEye as usize);
    let left_eye: (f64, f64) = (left_eye_f32.0 as f64, left_eye_f32.1 as f64);

    let right_eye_f32 = absolute_detection.keypoint(FaceIndex::RightEye as usize);
    let right_eye: (f64, f64) = (right_eye_f32.0 as f64, right_eye_f32.1 as f64);

    let roi = bbox_to_roi(
        face_detection.bbox(),
        image_size,
        Some(vec![left_eye, right_eye]),
        Some(ROI_SCALE),
        Some(SizeMode::SquareLong),
    );
    roi
}

pub(crate) struct FaceLandmark {
    model_path: PathBuf,
    model: FlatBufferModel,
}

/// `FaceLandmark` detection model as used by Google MediaPipe.
/// This model detects facial landmarks from a face image.
impl FaceLandmark {
    pub fn new(model_path: Option<String>) -> Result<FaceLandmark, Error> {
        let mut model_path_buf: PathBuf;

        if let Some(path) = model_path {
            model_path_buf = PathBuf::from(path);
        } else {
            model_path_buf =
                PathBuf::from("/home/tripg/Documents/repo/rs-face-detection-tflite/src/models/face_landmark.tflite");
        }
        let model = FlatBufferModel::build_from_file(model_path_buf.clone())?;

        Ok(FaceLandmark {
            model_path: model_path_buf,
            model,
        })
    }

    /// Run inference and return detections from a given image
    /// * Args:
    ///     - image (`Mat`): opencv mat.
    ///     - roi (`Option<Rect>`): Region within the image that contains a face.
    ///
    /// * Returns:
    ///     - (`Vec<Landmark>`) List of face landmarks in normalised coordinates relative to
    ///       the input image, i.e. values ranging from [0, 1].
    pub fn infer(&self, image: &Mat, roi: Option<Rect>) -> Result<Vec<Landmark>, Error> {
        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(&self.model, &resolver)?;
        let mut interpreter = builder.build()?;
        interpreter.allocate_tensors()?;

        let input_details = interpreter.get_input_details()?;
        let output_details = interpreter.get_output_details()?;

        let input_shape = input_details[0].dims.clone();
        let output_shape = output_details[0].dims.clone();

        let num_expected_elements = NUM_DIMS * NUM_LANDMARKS;
        if (output_shape[output_shape.len() - 1] as i32) < num_expected_elements {
            return Err(Error::msg(format!("incompatible model: {:?} < {:?}", output_shape, num_expected_elements)));
        };

        let (height, width) = (input_shape[1], input_shape[2]);
        let image_data = image_to_tensor(image, roi, Some((width as i32, height as i32)), false, (0., 1.), false)?;

        let input_data = image_data
            .tensor_data
            .into_dimensionality::<ndarray::IxDyn>()
            .unwrap()
            .insert_axis(Axis(0));

        // Infer model with input data
        let inputs = interpreter.inputs().to_vec();
        let input_index = inputs[0];
        let sub_tensor: Vec<f32> = input_data.into_iter().collect();
        interpreter
            .tensor_data_mut(input_index)?
            .copy_from_slice(&sub_tensor);
        interpreter.invoke()?;

        // retrieve outputs
        let outputs = interpreter.outputs().to_vec();
        let (data_index, raw_face_index) = (outputs[0], outputs[1]);

        // retrieve output info
        let data_info = interpreter
            .tensor_info(data_index)
            .ok_or(Error::msg("missing raw data outputs info"))?;

        let raw_face_info = interpreter
            .tensor_info(raw_face_index)
            .ok_or(Error::msg("missing raw face outputs info"))?;

        let raw_data_s: &[f32] = interpreter.tensor_data(data_index).unwrap();
        let raw_data: Array4<f32> = Array4::from_shape_vec(
            (data_info.dims[0], data_info.dims[1], data_info.dims[2], data_info.dims[3]),
            raw_data_s.to_vec(),
        )?;

        let raw_face_s: &[f32] = interpreter.tensor_data(raw_face_index).unwrap();
        let raw_face: Array4<f32> = Array4::from_shape_vec(
            (raw_face_info.dims[0], raw_face_info.dims[1], raw_face_info.dims[2], raw_face_info.dims[3]),
            raw_face_s.to_vec(),
        )?;

        let flatten = raw_face.mapv(|x| sigmoid(x)).flatten().to_vec();
        let face_flag = flatten[flatten.len() - 1];
        if face_flag <= DETECTION_THRESHOLD {
            return Ok(Vec::<Landmark>::new());
        };

        project_landmarks(
            raw_data,
            (width as i32, height as i32),
            image_data.original_size,
            image_data.padding,
            roi,
            false,
        )
    }
}

/// Convert face landmarks to render data.
/// This post-processing function can be used to generate a list of rendering
/// instructions from face landmark detection results.
/// * Args:
///     - face_landmarks (`Vec<Landmark>`): List of `Landmark` detection results returned by `FaceLandmark`.
///     - landmark_color (`Color`): Color of the individual landmark points.
///     - connection_color (1Color1): Color of the landmark connections that will be rendered as lines.
///     - thickness (float): Width of the lines and landmark point size in viewport units (e.g. pixels).
///     - output (list): Optional list of render annotations to add the items to.
///         If not provided, a new list will be created. Use this to add multiple landmark detections
///         into a single render annotation list.
///
/// * Returns:
///     - `Vec<Annotation>`: List of render annotations that should be rendered.
///       All positions are normalized, e.g. with a value range of [0, 1].
pub fn face_landmarks_to_render_data(
    face_landmarks: Vec<Landmark>, landmark_color: Color, connection_color: Color, thickness: Option<f32>,
    output: Option<Vec<Annotation>>,
) -> Vec<Annotation> {
    let thickness = thickness.unwrap_or(2.);
    let render_data = landmarks_to_render_data(
        face_landmarks,
        Vec::from(FACE_LANDMARK_CONNECTIONS),
        Some(landmark_color),
        Some(connection_color),
        Some(thickness),
        Some(true),
        output,
    );
    render_data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel};
    use crate::face_detection_lite::utils::convert_image_to_mat;
    use opencv::core::MatTraitConst;

    #[test]
    fn test_face_landmark() {
        let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();

        let im_bytes: &[u8] = include_bytes!("/home/tripg/Documents/face/datnt.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let img_shape = image.size().unwrap();

        let faces = face_detection.infer(&image, None).unwrap();

        let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();

        let face_landmark = FaceLandmark::new(None).unwrap();
        face_landmark.infer(&image, Some(face_roi));
    }
}

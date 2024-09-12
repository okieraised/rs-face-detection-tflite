use anyhow::Error;
use crate::face_detection_lite::render::{Annotation, AnnotationData, Color, Colors, landmarks_to_render_data, Point, RectOrOval};
use crate::face_detection_lite::transform::{bbox_from_landmarks, bbox_to_roi, SizeMode};
use crate::face_detection_lite::types::{Landmark, Rect};

/// Iris landmark detection model.
///
/// Ported from Google® MediaPipe (https://google.github.io/mediapipe/).
///
/// Model card:
///     https://mediapipe.page.link/iris-mc
///
/// Reference:
///     N/A


const MODEL_NAME: &str = "iris_landmark.tflite";
/// ROI scale factor for 25% margin around eye
const ROI_SCALE: (f64, f64) = (2.3, 2.3);
/// Landmark index of the left eye start point
const LEFT_EYE_START: usize = 33;
/// Landmark index of the left eye end point
const LEFT_EYE_END: usize = 133;
/// Landmark index of the right eye start point
const RIGHT_EYE_START: usize = 362;
/// Landmark index of the right eye end point
const RIGHT_EYE_END: usize = 263;
/// Number of face landmarks (from face landmark results)
const NUM_FACE_LANDMARKS: usize = 468;

/// Landmark element count (x, y, z)
const NUM_DIMS: i32 = 3;
const NUM_EYE_LANDMARKS: i32 = 71;
const NUM_IRIS_LANDMARKS: i32 = 5;

const EYE_LANDMARK_CONNECTIONS: [(i32, i32); 15] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (11, 12), (12, 13), (13, 14), (0, 9), (8, 14),
];

const MAX_EYE_LANDMARK: usize = EYE_LANDMARK_CONNECTIONS.len();

const LEFT_EYE_TO_FACE_LANDMARK_INDEX: [i32; 71] = [
    /// eye lower contour
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    /// eye upper contour excluding corners
    246, 161, 160, 159, 158, 157, 173,
    /// halo x2 lower contour
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    /// halo x2 upper contour excluding corners
    247, 30, 29, 27, 28, 56, 190,
    /// halo x3 lower contour
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    /// halo x3 upper contour excluding corners
    113, 225, 224, 223, 222, 221, 189,
    /// halo x4 upper contour (no upper due to mesh structure)
    /// or eyebrow inner contour
    35, 124, 46, 53, 52, 65,
    /// halo x5 lower contour
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    /// halo x5 upper contour excluding corners or eyebrow outer contour
    156, 70, 63, 105, 66, 107, 55, 193,
];

const RIGHT_EYE_TO_FACE_LANDMARK_INDEX: [i32; 71] = [
    /// eye lower contour
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    /// eye upper contour excluding corners
    466, 388, 387, 386, 385, 384, 398,
    /// halo x2 lower contour
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    /// halo x2 upper contour excluding corners
    467, 260, 259, 257, 258, 286, 414,
    /// halo x3 lower contour
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    /// halo x3 upper contour excluding corners
    342, 445, 444, 443, 442, 441, 413,
    /// halo x4 upper contour (no upper due to mesh structure)
    /// or eyebrow inner contour
    265, 353, 276, 283, 282, 295,
    /// halo x5 lower contour
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    /// halo x5 upper contour excluding corners or eyebrow outer contour
    383, 300, 293, 334, 296, 336, 285, 417,
];

// /// 35mm camera sensor diagonal (36mm * 24mm)
// const SENSOR_DIAGONAL_35MM: f64 = 1872_f64.sqrt();
/// average human iris size
const IRIS_SIZE_IN_MM: f64 = 11.8;

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IrisIndex {
    Center = 0,
    Left = 1,
    Top = 2,
    Right = 3,
    Bottom = 4,
}

/// Iris detection results.
/// contour data is 71 points defining the eye region
/// iris data is 5 keypoints
pub struct IrisResults {
    contour: Vec<Landmark>,
    iris: Vec<Landmark>,
}

impl IrisResults {
    pub fn new(contour: Vec<Landmark>, iris: Vec<Landmark>) -> Self {
        Self {
            contour,
            iris,
        }
    }

    pub fn eyeball_contour(&self) -> Vec<Landmark> {
        let lmk = self.contour.clone();
        lmk[0..MAX_EYE_LANDMARK].to_vec()
    }
}

/// Extract iris landmark detection ROIs from face landmarks.
/// Use this function to get the ROI for the left and right eye. The resulting
/// bounding boxes are suitable for passing to `IrisDetection` as the ROI
/// parameter. This is a pre-processing step originally found in the
/// MediaPipe sub-graph "iris_landmark_landmarks_to_roi":
/// * Args:
///     - face_landmarks (`Vec<Landmark>`): Result of a `FaceLandmark` call containing face
///             landmark detection results.
///     - image_size (`(i32, i32)`): Tuple of `(width, height)` representing the size
///             of the image in pixels. The image must be the same as the one used
///             for face lanmark detection.
///
/// * Return:
///     - (`(Rect, Rect)`) Tuple of ROIs containing the absolute pixel coordinates
///         of the left and right eye regions. The ROIs can be passed to
///         `IrisDetetion` together with the original image to detect iris
///         landmarks.
pub fn iris_roi_from_face_landmarks(face_landmarks: Vec<Landmark>, image_size: (i32, i32)) -> Result<(Rect, Rect), Error> {
    let left_eye_landmarks = [
        face_landmarks[LEFT_EYE_START],
        face_landmarks[LEFT_EYE_END],
    ];

    let bbox = bbox_from_landmarks(&left_eye_landmarks)?;
    let rotation_keypoints: Vec<(f64, f64)> = left_eye_landmarks.iter().map(|lmk| (lmk.x, lmk.y)).collect::<Vec<_>>();
    let (w, h) = image_size;
    let left_eye_roi = bbox_to_roi(
        bbox,
        (w, h),
        Some(rotation_keypoints),
        Some(ROI_SCALE),
        Some(SizeMode::SquareLong),
    )?;

    let right_eye_landmarks = [
        face_landmarks[RIGHT_EYE_START],
        face_landmarks[RIGHT_EYE_END],
    ];
    let bbox = bbox_from_landmarks(&right_eye_landmarks)?;
    let rotation_keypoints: Vec<(f64, f64)> = right_eye_landmarks.iter().map(|lmk| (lmk.x, lmk.y)).collect::<Vec<_>>();
    let right_eye_roi = bbox_to_roi(
        bbox,
        (w, h),
        Some(rotation_keypoints),
        Some(ROI_SCALE),
        Some(SizeMode::SquareLong),
    )?;

    Ok((left_eye_roi, right_eye_roi))
}

/// Convert eye contour to render data.
/// This post-processing function can be used to generate a list of rendering
/// instructions from iris landmark detection results.
/// * Args:
///     - eye_contour (`Vec<Landmark>`): List of `Landmark` detection results returned
///             by `IrisLandmark`.
///     - landmark_color (`Color`): Color of the individual landmark points.
///     - connection_color (`Color`): Color of the landmark connections that
///             will be rendered as lines.
///     - thickness (`f32`): Width of the lines and landmark point size in
///             viewport units (e.g. pixels).
///     - output (`Option<Vec<Annotation>>`): Optional list of render annotations to add the items
///             to. If not provided, a new list will be created.
///             Use this to add multiple landmark detections into a single render
///             annotation list.
/// * Returns:
///     - (`Vec<Annotation>`) List of render annotations that should be rendered.
///         All positions are normalized, e.g. with a value range of [0, 1].
pub fn eye_landmarks_to_render_data(
    eye_contour: Vec<Landmark>,
    landmark_color: Color,
    connection_color: Color,
    thickness: Option<f32>,
    output: Option<Vec<Annotation>>
) -> Vec<Annotation> {

    let thickness = thickness.unwrap_or(2.);
    let render_data = landmarks_to_render_data(
        eye_contour[0..MAX_EYE_LANDMARK].to_vec(),
        EYE_LANDMARK_CONNECTIONS.to_vec(),
        Some(landmark_color),
        Some(connection_color),
        Some(thickness),
        Some(true),
        output,
    );

    render_data
}

pub fn iris_landmarks_to_render_data(
    iris_landmarks: Vec<Landmark>,
    landmark_color: Option<Color>,
    oval_color: Option<Color>,
    thickness: Option<f64>,
    image_size: Option<(i32, i32)>,
    output: Option<Vec<Annotation>>,
) -> Result<Vec<Annotation>, Error> {
    let image_size = image_size.unwrap_or((-1, -1));
    let thickness = thickness.unwrap_or(1.0);

    let mut annotations: Vec<Annotation> = Vec::new();

    if let Some(oval_color) = oval_color {
        let iris_radius = get_iris_diameter(&iris_landmarks, image_size) / 2.0;
        let (width, height) = image_size;
        if width < 2 ||  height < 2 {
            return Err(Error::msg("oval_color requires a valid image_size arg"))
        }
        let radius_h = iris_radius / width as f64;
        let radius_v = iris_radius / height as f64;
        let iris_center = iris_landmarks[IrisIndex::Center as usize];
        let oval = RectOrOval::new(
            iris_center.x - radius_h, iris_center.y - radius_v,
            iris_center.x + radius_h, iris_center.y + radius_v,
            true,
        );
        annotations.push(Annotation::new(
            vec![AnnotationData::RectOrOval(oval)],
            true,
            thickness,
            oval_color,
        ))
    }

    if let Some(landmark_color) = landmark_color {
        let points: Vec<Point> = iris_landmarks.iter().map(|lmk| Point::new(lmk.x, lmk.y)).collect::<Vec<Point>>();
        let data: Vec<AnnotationData> = points.iter().map(|&p| AnnotationData::Point(p)).collect::<Vec<AnnotationData>>();

        annotations.push(Annotation::new(
            data,
            true,
            thickness,
            landmark_color,
        ))
    }

    if let Some(mut output_vec) = output {
        output_vec.extend(annotations.clone());
        Ok(output_vec.clone())
    } else {
        Ok(annotations)
    }
}

/// Calculate the iris diameter in pixels
fn get_iris_diameter(
    iris_landmarks: &Vec<Landmark>,
    image_size: (i32, i32),
) -> f64 {
    let (width, height) = image_size;

    let get_landmark_depth = |a: Landmark, b: Landmark| -> f64 {
        let (x0, y0, x1, y1) = (a.x * width as f64, a.y * height as f64, b.x * width as f64, b.y * height as f64);

        let res: f64 = (x0 - x1).powi(2) + (y0 - y1).powi(2);
        res.sqrt()
    };

    let iris_size_horiz = get_landmark_depth(
        iris_landmarks[IrisIndex::Left as usize],
        iris_landmarks[IrisIndex::Right as usize]);

    let iris_size_vert = get_landmark_depth(
        iris_landmarks[IrisIndex::Top as usize],
        iris_landmarks[IrisIndex::Bottom as usize],
    );

    (iris_size_vert + iris_size_horiz) / 2.
}

/// Calculate iris depth in mm from landmarks and lens focal length in mm
fn get_iris_depth(
    iris_landmarks: Vec<Landmark>,
    focal_length_mm: f64,
    iris_size_px: f64,
    image_size: (i32, i32),
) -> f64 {
    let (width, height) = image_size;
    let center = iris_landmarks[IrisIndex::Center as usize];
    let (x0, y0) = (width / 2, height / 2);
    let (x1, y1) = (center.x * width as f64, center.y * height as f64);
    let y_square: f64 = (x0 as f64 - x1).powi(2) + (y0 as f64 - y1).powi(2);
    let y = y_square.sqrt();
    let x_square: f64 = focal_length_mm.powi(2) + y.powi(2);
    let x = x_square.sqrt();
    IRIS_SIZE_IN_MM * x / iris_size_px
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel};
    use crate::face_detection_lite::utils::convert_image_to_mat;
    use opencv::core::MatTraitConst;
    use crate::face_detection_lite::face_landmark::{face_detection_to_roi, FaceLandmark};

    #[test]
    fn test_face_landmark() {
        let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();
        let im_bytes: &[u8] = include_bytes!("/home/tripg/Documents/face/datnt.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let img_shape = image.size().unwrap();
        let faces = face_detection.infer(&image, None).unwrap();
        let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();
        let face_landmark = FaceLandmark::new(None).unwrap();
        let lmks = face_landmark.infer(&image, Some(face_roi)).unwrap();

        iris_roi_from_face_landmarks(lmks, (img_shape.width, img_shape.height));


    }
}
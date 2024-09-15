use crate::face_detection_lite::nms::non_maximum_suppression;
use crate::face_detection_lite::transform::{detection_letterbox_removal, image_to_tensor, sigmoid};
use crate::face_detection_lite::types::{Detection, Rect};
use crate::face_detection_lite::utils::convert_image_to_mat;
use anyhow::Error;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Array2, Array3, Axis};
use opencv::core::Mat;
use std::ops::{AddAssign, Div};
use std::path::PathBuf;
use tflite::op_resolver::OpResolver;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};

/// BlazeFace face detection.
///
/// Ported from Google® MediaPipe (https://google.github.io/mediapipe/).
///
/// Model card:
///     https://mediapipe.page.link/blazeface-mc
///
/// Reference:
///     V. Bazarevsky et al. BlazeFace: Sub-millisecond
///     Neural Face Detection on Mobile GPUs. CVPR
///     Workshop on Computer Vision for Augmented and
///     Virtual Reality, Long Beach, CA, USA, 2019.
///

pub struct SSDOptions {
    pub num_layers: i32,
    pub input_size_height: i32,
    pub input_size_width: i32,
    pub anchor_offset_x: f32,
    pub anchor_offset_y: f32,
    pub strides: Vec<i32>,
    pub interpolated_scale_aspect_ratio: f32,
}

impl SSDOptions {
    pub fn new_front() -> Self {
        Self {
            num_layers: 4,
            input_size_height: 128,
            input_size_width: 128,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![8, 16, 16, 16],
            interpolated_scale_aspect_ratio: 1.0,
        }
    }

    pub fn new_back() -> Self {
        Self {
            num_layers: 4,
            input_size_height: 256,
            input_size_width: 256,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![16, 32, 32, 32],
            interpolated_scale_aspect_ratio: 1.0,
        }
    }

    pub fn new_short() -> Self {
        Self {
            num_layers: 4,
            input_size_height: 128,
            input_size_width: 128,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![8, 16, 16, 16],
            interpolated_scale_aspect_ratio: 1.0,
        }
    }

    pub fn new_full() -> Self {
        Self {
            num_layers: 1,
            input_size_height: 192,
            input_size_width: 192,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![4, 0, 0, 0],
            interpolated_scale_aspect_ratio: 0.0,
        }
    }
}

/// Indexes of keypoints returned by the face detection model.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FaceIndex {
    LeftEye = 0,
    RightEye = 1,
    NoseTip = 2,
    Mouth = 3,
    LeftEyeTragion = 4,
    RightEyeTragion = 5,
}

impl TryFrom<i32> for FaceIndex {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            x if x == FaceIndex::LeftEye as i32 => Ok(FaceIndex::LeftEye),
            x if x == FaceIndex::RightEye as i32 => Ok(FaceIndex::RightEye),
            x if x == FaceIndex::NoseTip as i32 => Ok(FaceIndex::NoseTip),
            x if x == FaceIndex::Mouth as i32 => Ok(FaceIndex::Mouth),
            x if x == FaceIndex::LeftEyeTragion as i32 => Ok(FaceIndex::LeftEyeTragion),
            x if x == FaceIndex::RightEyeTragion as i32 => Ok(FaceIndex::RightEyeTragion),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaceDetectionModel {
    FrontCamera = 0,
    BackCamera = 1,
    Short = 2,
    Full = 3,
    FullSparse = 4,
}

const MODEL_NAME_BACK: &str = "face_detection_back.tflite";
const MODEL_NAME_FRONT: &str = "face_detection_front.tflite";
const MODEL_NAME_SHORT: &str = "face_detection_short_range.tflite";
const MODEL_NAME_FULL: &str = "face_detection_full_range.tflite";
const MODEL_NAME_FULL_SPARSE: &str = "face_detection_full_range_sparse.tflite";

/// score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
/// this lower limit is safe for use with the sigmoid functions and float32
const RAW_SCORE_LIMIT: f32 = 80.0;

/// threshold for confidence scores
const MIN_SCORE: f32 = 0.5;

/// NMS similarity threshold
const MIN_SUPPRESSION_THRESHOLD: f32 = 0.3;

/// BlazeFace face detection model as used by Google MediaPipe.
/// This model can detect multiple faces and returns a list of detections.
/// Each detection contains the normalised [0,1] position and size of the
/// detected face, as well as a number of keypoints (also normalised to
/// [0,1]).
pub(crate) struct FaceDetection {
    model_path: PathBuf,
    model: FlatBufferModel,
    anchors: Array2<f32>,
}

impl FaceDetection {
    pub fn new(model_type: FaceDetectionModel, model_path: Option<String>) -> Result<FaceDetection, Error> {
        let mut ssd_opts = SSDOptions::new_front(); // Placeholder for SSD options
        let mut model_path_buf: PathBuf;

        if let Some(path) = model_path {
            model_path_buf = PathBuf::from(path);
        } else {
            model_path_buf = PathBuf::from("./models");
        }

        match model_type {
            FaceDetectionModel::FrontCamera => {
                model_path_buf.push(MODEL_NAME_FRONT);
                ssd_opts = SSDOptions::new_front();
            }
            FaceDetectionModel::BackCamera => {
                model_path_buf.push(MODEL_NAME_BACK);
                ssd_opts = SSDOptions::new_back();
            }
            FaceDetectionModel::Short => {
                model_path_buf.push(MODEL_NAME_SHORT);
                ssd_opts = SSDOptions::new_short();
            }
            FaceDetectionModel::Full => {
                model_path_buf.push(MODEL_NAME_FULL);
                ssd_opts = SSDOptions::new_full();
            }
            FaceDetectionModel::FullSparse => {
                model_path_buf.push(MODEL_NAME_FULL_SPARSE);
                ssd_opts = SSDOptions::new_full();
            }
            _ => return Err(Error::msg("unsupported model type")),
        }

        let anchors = ssd_generate_anchors(&ssd_opts);
        let model = FlatBufferModel::build_from_file(model_path_buf.clone())?;

        Ok(FaceDetection {
            model_path: model_path_buf,
            model,
            anchors,
        })
    }

    /// Run inference and return detections from a given image
    /// * Args:
    ///     - image (`Mat`): OpenCV matrix.
    ///     - roi (`Rect`): Optional region within the image that may
    ///                 contain faces.
    ///
    /// * Returns:
    ///     (`Vec<Detection>`) List of detection results with relative coordinates.
    pub fn infer(&self, image: &Mat, roi: Option<Rect>) -> Result<Vec<Detection>, Error> {
        // Init model interpreter
        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(&self.model, &resolver)?;
        let mut interpreter = builder.build()?;
        interpreter.allocate_tensors()?;

        // Get model input image shape
        let input_details = interpreter.get_input_details()?;
        let input_shape = input_details[0].dims.clone();

        // convert image to opencv matrix
        let (height, width) = (input_shape[1], input_shape[2]);

        let image_data = image_to_tensor(&image, roi, Some((width as i32, height as i32)), true, (-1.0, 1.0), false)?;

        // Add additional axis
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
        let (bbox_index, score_index) = (outputs[0], outputs[1]);

        // retrieve output info
        let bbox_info = interpreter
            .tensor_info(bbox_index)
            .ok_or(Error::msg("missing bounding box outputs info"))?;

        let score_info = interpreter
            .tensor_info(score_index)
            .ok_or(Error::msg("missing score outputs info"))?;

        // Retrieve raw detection boxes and convert to Array3<f32>
        let raw_boxes_s: &[f32] = interpreter.tensor_data(bbox_index).unwrap();
        let raw_boxes: Array3<f32> =
            Array3::from_shape_vec((bbox_info.dims[0], bbox_info.dims[1], bbox_info.dims[2]), raw_boxes_s.to_vec())?;

        let raw_scores_s: &[f32] = interpreter.tensor_data(score_index).unwrap();
        let raw_scores =
            Array::from_shape_vec((score_info.dims[0], score_info.dims[1], score_info.dims[2]), raw_scores_s.to_vec())?;

        let boxes = self.decode_boxes(raw_boxes, input_shape[1] as f32)?;
        let scores = self.get_sigmoid_score(raw_scores)?;

        let detections = self.convert_to_detections(boxes, scores)?;
        let pruned_detections = non_maximum_suppression(detections, MIN_SUPPRESSION_THRESHOLD, Some(MIN_SCORE), true);

        let detections = detection_letterbox_removal(pruned_detections, image_data.padding);
        // println!("detections: {:?}", detections);
        Ok(detections)
    }

    fn decode_boxes(&self, raw_boxes: Array3<f32>, scale: f32) -> Result<Array3<f32>, Error> {
        let shape = raw_boxes.shape();
        let num_points = shape[shape.len() - 1] / 2;

        let shape = (raw_boxes.len() / (num_points * 2), num_points, 2);
        let mut boxes = raw_boxes.mapv(|x| x / scale).to_shape(shape)?.to_owned();

        let mut boxes_0 = boxes.slice_mut(s![.., 0, ..]);
        boxes_0 += &self.anchors;

        for i in 2..num_points {
            let mut boxes_slice = boxes.slice_mut(s![.., i, ..]);
            boxes_slice += &self.anchors;
        }

        let center = boxes.slice(s![.., 0, ..]).to_owned();
        let half_size = &boxes.slice(s![.., 1, ..]) / 2.0;
        {
            let mut boxes_0 = boxes.slice_mut(s![.., 0, ..]);
            boxes_0.assign(&(&center - &half_size));
        }
        {
            let mut boxes_1 = boxes.slice_mut(s![.., 1, ..]);
            boxes_1.assign(&(&center + &half_size));
        }
        // println!("boxes: {:?}", boxes);
        Ok(boxes)
    }

    /// Extracted loop from ProcessCPU (line 327) in
    /// mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    fn get_sigmoid_score(&self, mut raw_scores: Array3<f32>) -> Result<Array3<f32>, Error> {
        raw_scores.par_mapv_inplace(|x| {
            if x < -RAW_SCORE_LIMIT {
                -RAW_SCORE_LIMIT
            } else if x > RAW_SCORE_LIMIT {
                RAW_SCORE_LIMIT
            } else {
                x
            }
        });

        raw_scores.par_mapv_inplace(|x| sigmoid(x));

        Ok(raw_scores)
    }

    /// Apply detection threshold, filter invalid boxes and return detection instance.
    fn convert_to_detections(&self, boxes: Array3<f32>, scores: Array3<f32>) -> Result<Vec<Detection>, Error> {
        fn is_valid(bbox: &Array2<f32>) -> bool {
            let row0 = bbox.row(0);
            let row1 = bbox.row(1);

            row1.iter().zip(row0.iter()).all(|(&x1, &x0)| x1 > x0)
        }

        let mut detections: Vec<Detection> = Vec::new();
        let score_above_threshold = scores.mapv(|score| score > MIN_SCORE);
        let mut filtered_scores_idx: Vec<usize> = Vec::new();
        for ((_, j, _), &is_true) in score_above_threshold.indexed_iter() {
            if is_true {
                filtered_scores_idx.push(j);
            }
        }

        // Filter scores
        let mut filtered_scores = Vec::new();
        for ((i, j, k), &is_true) in score_above_threshold.indexed_iter() {
            if is_true {
                filtered_scores.push(scores[[i, j, k]]);
            }
        }

        let mut score_idx: Vec<usize> = Vec::new();
        for ((i, j, k), &is_above_threshold) in score_above_threshold.indexed_iter() {
            if is_above_threshold {
                score_idx.push(j);
            }
        }

        // Filter bounding boxes
        let mut filtered_boxes = Vec::new();
        for &i in &score_idx {
            filtered_boxes.push(boxes.slice(s![i, .., ..]).to_owned());
        }

        for (bbox, score) in filtered_boxes.into_iter().zip(filtered_scores) {
            if is_valid(&bbox) {
                let (data, _) = bbox.into_raw_vec_and_offset();
                detections.push(Detection::new(data, score.into()))
            }
        }
        Ok(detections)
    }
}

/// (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
fn ssd_generate_anchors(opts: &SSDOptions) -> Array2<f32> {
    let mut layer_id = 0;
    let num_layers = opts.num_layers;
    let strides = &opts.strides;
    let input_height = opts.input_size_height;
    let input_width = opts.input_size_width;
    let anchor_offset_x = opts.anchor_offset_x;
    let anchor_offset_y = opts.anchor_offset_y;
    let interpolated_scale_aspect_ratio = opts.interpolated_scale_aspect_ratio;
    let mut anchors = Vec::new();

    while layer_id < num_layers {
        let mut last_same_stride_layer = layer_id;
        let mut repeats = 0;

        while last_same_stride_layer < num_layers
            && strides[last_same_stride_layer as usize] == strides[layer_id as usize]
        {
            last_same_stride_layer += 1;
            // aspect_ratios are added twice per iteration
            repeats += if interpolated_scale_aspect_ratio == 1.0 { 2 } else { 1 };
        }

        let stride = strides[layer_id as usize];
        let feature_map_height = input_height / stride;
        let feature_map_width = input_width / stride;

        for y in 0..feature_map_height {
            let y_center = (y as f32 + anchor_offset_y) / feature_map_height as f32;
            for x in 0..feature_map_width {
                let x_center = (x as f32 + anchor_offset_x) / feature_map_width as f32;
                for _ in 0..repeats {
                    anchors.push((x_center, y_center));
                }
            }
        }
        layer_id = last_same_stride_layer;
    }

    let num_anchors = anchors.len();
    let mut anchors_array = Array2::<f32>::zeros((num_anchors, 2));

    for (i, (x, y)) in anchors.into_iter().enumerate() {
        anchors_array[[i, 0]] = x;
        anchors_array[[i, 1]] = y;
    }
    anchors_array
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detection_lite::render::render_to_image;

    // #[test]
    // fn test_face_detection() {
    //     let face_detection = match FaceDetection::new(FaceDetectionModel::BackCamera, None) {
    //         Ok(face_detection) => face_detection,
    //         Err(e) => {
    //             println!("{:?}", e);
    //             return;
    //         }
    //     };
    //
    //     let im_bytes: &[u8] = include_bytes!("/Users/tripham/Desktop/face-detection-tflite/women.png");
    //     let image = convert_image_to_mat(im_bytes).unwrap();
    //     let detections = face_detection.infer(&image, None).unwrap();
    // }

    #[test]
    fn test_ndarray() {
        let test_arr: Array2<f32> = Array2::<f32>::from(vec![
            [0.24733608, 0.31957608],
            [0.74665025, 0.76793867],
            [0.39087012, 0.44037619],
            [0.60614576, 0.43600613],
            [0.5057843, 0.54600358],
            [0.50618416, 0.64536804],
            [0.27929265, 0.49235079],
            [0.71279182, 0.48242962],
        ]);
        println!("test_arr: {:?}", test_arr);

        let mut res: Vec<(f32, f32)> = Vec::new();

        for row in 0..test_arr.nrows() {
            let x = test_arr.row(row);
            res.push((x[0], x[1]));
        }
        println!("res: {:?}", res);
    }
}

use std::ops::{AddAssign, Div};
use std::path::PathBuf;
use anyhow::Error;
use bytemuck::cast_slice;
use ndarray::{s, Array, Array2, Array3, Axis, IxDyn};
use opencv::imgcodecs::imwrite;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};
use crate::face_detection_lite::transform::image_to_tensor;
use crate::face_detection_lite::utils::convert_image_to_mat;

pub struct SSDOptions{
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

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FaceIndex {
    LeftEye = 0,
    RightEye = 1,
    NoseTip = 2,
    MOUTH = 3,
    LeftEyeTragion = 4,
    RightEyeTragion = 5,
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




pub struct FaceDetection {
    // model_path: PathBuf,
    // interpreter: Interpreter,
    // input_index: usize,
    input_shape: Vec<usize>,
    // bbox_index: usize,
    // score_index: usize,
    anchors: Array2<f32>,
}

impl FaceDetection {
    pub fn new(model_type: FaceDetectionModel, model_path: Option<String>) -> Result<FaceDetection, Error> {
        let mut ssd_opts = SSDOptions::new_front();  // Placeholder for SSD options
        let mut model_path_buf: PathBuf;

        if let Some(path) = model_path {
            model_path_buf = PathBuf::from(path);
        } else {
            model_path_buf = PathBuf::from("/Users/tripham/Desktop/rs-face-detection-tflite/src/models");
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

        println!("model_path_buf: {:?}", model_path_buf.as_path());

        let model = FlatBufferModel::build_from_file(model_path_buf)?;
        let resolver = BuiltinOpResolver::default();

        let builder = InterpreterBuilder::new(&model, &resolver)?;
        let mut interpreter = builder.build()?;


        let input_details = interpreter.get_input_details()?;
        let output_details = interpreter.get_output_details()?;
        let anchors = ssd_generate_anchors(&ssd_opts);


        let input_shape = input_details[0].dims.clone();

        println!("input_details: {:?}", input_details);
        println!("output_details: {:?}", output_details);
        // println!("anchors: {:?}", anchors);


        /// process here
        let height = input_shape[1];
        let width = input_shape[2];

        let im_bytes: &[u8] = include_bytes!("/Users/tripham/Desktop/face-detection-tflite/women.jpeg");
        let image = convert_image_to_mat(im_bytes)?;

        // imwrite("./test.jpeg", &image, &Default::default());

        let image_data = image_to_tensor(
            &image,
            None,
            Some((width as f32, height as f32)),
            true,
            (-1.0, 1.0),
            false
        )?;

        // Add additional axis
        let input_data = image_data.tensor_data.
            into_dimensionality::<ndarray::IxDyn>().
            unwrap().
            insert_axis(ndarray::Axis(0));

        let inputs = interpreter.inputs().to_vec();
        println!("{:?}", inputs);

        let input_index = inputs[0];
        let info = interpreter.tensor_info(input_index).unwrap();
        println!("tensor info: {:?}", info);

        let sub_tensor: Vec<f32> = input_data.into_iter().collect();

        interpreter.tensor_data_mut(input_index)?.copy_from_slice(&sub_tensor);
        interpreter.invoke()?;

        let outputs = interpreter.outputs().to_vec();
        println!("{:?}", outputs);

        let bbox_index = outputs[0];
        let score_index = outputs[1];

        println!("bbox_index: {:?}", bbox_index);

        let bbox_info = interpreter.tensor_info(bbox_index).unwrap();
        println!("bbox_info: {:?}", bbox_info);

        let raw_boxes: &[f32] = interpreter.tensor_data(bbox_index).unwrap();
        // println!("raw_boxes {:?}", raw_boxes);

        let mut dimensions: Vec<usize> =  Vec::with_capacity(bbox_info.dims.len());
        for dim in &bbox_info.dims {
            dimensions.push(*dim);
        }
        let arr_raw_boxes: Array3<f32> = Array3::from_shape_vec(
            (dimensions[0], dimensions[1], dimensions[2]),
            raw_boxes.to_vec(),
        )?;
        println!("arr_raw_boxes {:?}", arr_raw_boxes);

        // let score_info = interpreter.tensor_info(score_index).unwrap();
        // println!("score_info: {:?}", score_info);
        //
        // let raw_score: &[f32] = interpreter.tensor_data(score_index).unwrap();
        // // println!("raw_score {:?}", raw_score);
        //
        // let mut dimensions: Vec<usize> =  Vec::with_capacity(score_info.dims.len());
        // for dim in &score_info.dims {
        //     dimensions.push(*dim);
        // }
        // let arr_raw_score = Array::from_shape_vec(
        //     dimensions,
        //     raw_score.to_vec(),
        // )?;
        // println!("arr_raw_score {:?}", arr_raw_score);

        // decode_boxes
        // let scale = input_shape[1];
        // let shape = arr_raw_boxes.shape();
        // let num_points = shape[shape.len() - 1] / 2;
        // println!("scale {:?}", scale);
        // println!("shape {:?}", shape);
        // println!("num_points {:?}", num_points);
        //
        // let raw_boxes_3: Array3<f32> = arr_raw_boxes.clone().into_shape_with_order([arr_raw_boxes.shape()[0].clone(), arr_raw_boxes.shape()[1].clone(), arr_raw_boxes.shape()[2].clone()])?;
        // println!("raw_boxes_3 {:?}", raw_boxes_3);
        // let transposed_tensors: Array3<f32> = arr_raw_boxes.permuted_axes([shape[0], num_points, 2]).try_into()?;
        // println!("transposed_tensors {:?}", transposed_tensors);


        Ok(FaceDetection {
            input_shape,
            anchors,
        })
    }

    fn decode_boxes(&self, raw_boxes: Array<f32, IxDyn>) -> Result<(), Error> {

        let scale = self.input_shape[1];
        let shape = raw_boxes.shape();
        let num_points = shape[shape.len() - 1] / 2;

        println!("num_points {:?}", num_points);

        // // Reshape raw_boxes to (-1, num_points, 2)
        // let reshaped_boxes = raw_boxes.to_shape((shape[0], num_points, 2))?;
        //
        // // Scale all values (applies to positions, width, and height alike)
        // let mut boxes = reshaped_boxes / scale;
        //
        // // Adjust center coordinates (x, y) to anchor positions
        // // Anchors must be a 2D array matching (num_boxes, 2)
        // for i in 0..self.anchors.shape()[0] {
        //     let anchor = self.anchors.row(i).to_owned();
        //
        //     // Apply anchors to box coordinates
        //     boxes.slice_mut(s![i, 0, ..]).add_assign(&anchor);
        //
        //     // Apply anchor for remaining points in the box
        //     for j in 2..num_points {
        //         boxes.slice_mut(s![i, j, ..]).add_assign(&anchor);
        //     }
        // }
        //
        // // Convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        // let center = boxes.index_axis(Axis(1), 0).to_owned();
        // let half_size = boxes.index_axis(Axis(1), 1).div(2.0);
        //
        // // Modify the boxes to (xmin, ymin) and (xmax, ymax)
        // let mut xmin_ymin = boxes.index_axis_mut(Axis(1), 0);
        // let mut xmax_ymax = boxes.index_axis_mut(Axis(1), 1);
        //
        // xmin_ymin -= &half_size;
        // xmax_ymax += &half_size;
        //
        // Ok(boxes)
        Ok(())
    }
}

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

        while last_same_stride_layer < num_layers && strides[last_same_stride_layer as usize] == strides[layer_id as usize] {
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

    #[test]
    fn test_face_detection() {
        let model = match FaceDetection::new(FaceDetectionModel::BackCamera, None) {
            Ok(model) => {model}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };
    }
}
use std::f64::EPSILON;
use ndarray::{Array2, ArrayD, Axis, Zip};
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::NdFloat;
use image::{DynamicImage, ImageBuffer, RgbImage, RgbaImage, GenericImageView, open};
use std::path::Path;
use anyhow::Error;
use ndarray::{Array3, Array};
use opencv::{core, imgproc, prelude::*};
use opencv::core::{copy_make_border, flip, Point2f, Scalar, Size, Vec3b, Vector, BORDER_CONSTANT};
use opencv::imgcodecs::imwrite;
use opencv::imgproc::{resize, INTER_LINEAR};
use crate::face_detection_lite::types::{BBox, Detection, ImageTensor, Landmark, Rect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeMode {
    /// Keep width and height as calculated.
    Default = 0,

    /// Make square using `max(width, height)`.
    SquareLong = 1,

    /// Make square using `min(width, height)`.
    SquareShort = 2,
}

impl From<i32> for SizeMode {
    fn from(value: i32) -> Self {
        match value {
            1 => SizeMode::SquareLong,
            2 => SizeMode::SquareShort,
            _ => SizeMode::Default,
        }
    }
}

impl SizeMode {
    pub fn to_int(self) -> i32 {
        self as i32
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn detection_letterbox_removal(detections: Vec<Detection>, padding: (f64, f64, f64, f64)) -> Vec<Detection> {
    let (left, top, right, bottom) = padding;
    let h_scale = 1.0 - (left + right);
    let v_scale = 1.0 - (top + bottom);

    // Ensure we are not dividing by very small values
    assert!(h_scale > EPSILON, "Horizontal scale is too small");
    assert!(v_scale > EPSILON, "Vertical scale is too small");

    // Adjust a detection
    fn adjust_data(det: &Detection, left: f32, top: f32, h_scale: f32, v_scale: f32) -> Detection {
        let mut adjusted_data = det.data.clone();
        for mut row in adjusted_data.rows_mut() {
            row[0] = (row[0] - left) / h_scale;
            row[1] = (row[1] - top) / v_scale;
        }

        let (adjusted, _) = adjusted_data.into_raw_vec_and_offset();

        Detection::new(adjusted, det.score)
    }

    // Apply adjustment to all detections
    detections
        .into_iter()
        .map(|detection| adjust_data(
            &detection,
            left as f32,
            top as f32,
            h_scale as f32,
            v_scale as f32,
        ))
        .collect()
}


pub fn bbox_from_landmarks(landmarks: &[Landmark]) -> Result<BBox, Error> {
    if landmarks.len() < 2 {
        return Err(Error::msg("landmarks must contain at least 2 items"))
    }

    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for landmark in landmarks {
        let (x, y) = (landmark.x, landmark.y);
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }

    Ok(BBox {
        xmin,
        ymin,
        xmax,
        ymax,
    })
}

pub fn image_to_tensor(
    image: &Mat,
    roi: Option<Rect>,
    output_size: Option<(i32, i32)>,
    keep_aspect_ratio: bool,
    output_range: (f64, f64),
    flip_horizontal: bool,
) -> Result<ImageTensor, Error> {

    let original_img_shape = image.size()?;
    let mut roi = roi.unwrap_or_else(|| Rect {
        x_center: 0.5,
        y_center: 0.5,
        width: 1.0,
        height: 1.0,
        rotation: 0.0,
        normalized: true,
    });

    roi = roi.scaled((original_img_shape.width as f64, original_img_shape.height as f64), false);

    let output_size = output_size.unwrap_or((roi.width as i32, roi.height as i32));

    let (width, height) = if keep_aspect_ratio {
        (roi.size().0 as i32, roi.size().1 as i32)
    } else {
        output_size
    };

    // Define the corresponding points in the input image
    let mut src_points = Vector::<Point2f>::new();
    for (_, &point) in roi.points().iter().enumerate() {
        src_points.push(Point2f::new(point.0.clone() as f32, point.1.clone() as f32));
    }

    // Define the corresponding points in the output image
    let mut dst_points = Vector::<Point2f>::new();
    dst_points.push(Point2f::new(0.0, 0.0));
    dst_points.push(Point2f::new(width as f32, 0.));
    dst_points.push(Point2f::new(width as f32, height as f32));
    dst_points.push(Point2f::new(0., height as f32));

    let transformation_matrix = imgproc::get_perspective_transform(&src_points, &dst_points, INTER_LINEAR)?;

    // Perform the perspective warp
    let mut roi_image = Mat::default();
    imgproc::warp_perspective(
        image,
        &mut roi_image,
        &transformation_matrix,
        Size::new(width, height),
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    // imwrite("./perspective_image.jpeg", &roi_image, &Default::default())?;

    let mut pad_x: f64 = 0.;
    let mut pad_y: f64 = 0.;

    if keep_aspect_ratio {
        let out_aspect = (output_size.1 / output_size.0) as f64;
        let roi_aspect = roi.height / roi.width;
        let (mut new_width, mut new_height) = (roi.width as i32, roi.height as i32);

        if out_aspect > roi_aspect {
            new_height = (roi.width * out_aspect) as i32;
            pad_y = (1.0 - roi_aspect / out_aspect) / 2.0;
        } else {
            new_width = (roi.height / out_aspect) as i32;
            pad_x = (1.0 - out_aspect / roi_aspect) / 2.0;
        }

        if new_width != roi.width as i32 || new_height != roi.height as i32 {
            let (pad_h, pad_v) = ((pad_x * new_width as f64) as i32, (pad_y * new_height as f64) as i32);
            let top = pad_v;
            let bottom = pad_v;
            let left = pad_h;
            let right = pad_h;

            let mut padded_image = Mat::default();
            copy_make_border(
                &roi_image,
                &mut padded_image,
                top, bottom, left, right,
                BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
            //imwrite("./padded_image.jpeg", &padded_image, &Default::default())?;

            let mut resized_image = Mat::default();
            resize(
                &padded_image,
                &mut resized_image,
                Size::new(new_width, new_height),
                0.0, 0.0,
                INTER_LINEAR,
            )?;
            // imwrite("./resized_image.jpeg", &resized_image, &Default::default())?;
            roi_image = resized_image;
        }

        let mut resized_image = Mat::default();
        resize(
            &roi_image,
            &mut resized_image,
            Size::new(output_size.0, output_size.1),
            0.0,
            0.0,
            INTER_LINEAR,
        ).unwrap();

        // imwrite("./resized_image2.jpeg", &resized_image, &Default::default())?;
        roi_image = resized_image;
    }

    if flip_horizontal {
        let mut flipped_image = Mat::default();
        flip(&roi_image, &mut flipped_image, 1)?;
        roi_image = flipped_image;
    };

    let min_val = output_range.0;
    let max_val = output_range.1;
    let img_shape = roi_image.size()?;

    let mut tensors = Array3::<f32>::zeros((
        img_shape.width as usize,
        img_shape.height as usize,
        3usize,
    ));

    for i in 0..3 {
        for y in 0..img_shape.width as usize {
            for x in 0..img_shape.height as usize {
                let pixel_value = roi_image.at_2d::<Vec3b>(y as i32, x as i32).unwrap()[i];
                tensors[[y, x, i]] = (pixel_value as f64 * (max_val - min_val) / 255.0 + min_val) as f32;
            }
        }
    }
    let tensor_data = tensors.into_dyn();

    Ok(ImageTensor {
        tensor_data,
        padding: (pad_x, pad_y, pad_x, pad_y),
        original_size: (original_img_shape.width, original_img_shape.height),
    })
}

fn vec_to_array2(vec: Vec<(f32, f32)>) -> Array2<f32> {
    // Flatten the Vec<(f32, f32)> into a Vec<f32>
    let flattened: Vec<f32> = vec.into_iter().flat_map(|(x, y)| vec![x, y]).collect();

    // Get the number of rows, which is the number of elements in the original Vec
    let rows = flattened.len() / 2;

    // Create a 2D array with shape (rows, 2)
    Array2::from_shape_vec((rows, 2), flattened).unwrap()
}

fn perspective_transform_coeff(
    src_points: &Vec<(f32, f32)>,
    dst_points: &Vec<(f32, f32)>,
) -> Result<[f32; 8], Error> {
    // Ensure that there are exactly 4 source and 4 destination points
    if src_points.len() != 4 || dst_points.len() != 4 {
        return Err(Error::msg("src_points and dst_points must contain exactly 4 points each."));
    }

    let mut matrix = Vec::with_capacity(8 * 8);
    let mut b = Vec::with_capacity(8);

    // Construct the matrix and vector B for solving
    for ((x, y), (X, Y)) in dst_points.iter().zip(src_points.iter()) {
        matrix.push(vec![*x, *y, 1.0, 0.0, 0.0, 0.0, -X * x, -X * y]);
        matrix.push(vec![0.0, 0.0, 0.0, *x, *y, 1.0, -Y * x, -Y * y]);
        b.push(*X);
        b.push(*Y);
    }

    // Flatten the matrix into a single vector
    let flat_matrix: Vec<f32> = matrix.into_iter().flatten().collect();

    // Convert the flattened matrix to a 2D array (8x8) and B to a 1D array (8)
    let a = nalgebra::DMatrix::from_row_slice(8, 8, &flat_matrix);
    let b = nalgebra::DVector::from_column_slice(&b);

    // Solve the linear system A * coeffs = B
    let coeffs = a.lu().solve(&b);

    match coeffs {
        Some(solution) => {
            // Convert solution to an array of size 8
            let mut result = [0.0; 8];
            for i in 0..8 {
                result[i] = solution[i];
            }
            Ok(result)
        }
        None => Err(Error::msg("Failed to solve the linear system.")),
    }
}






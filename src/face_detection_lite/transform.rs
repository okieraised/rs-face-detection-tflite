use std::default;
use crate::face_detection_lite::types::{BBox, Detection, ImageTensor, Landmark, Rect};
use anyhow::Error;
use image::GenericImageView;
use ndarray::Array3;
use ndarray::{s, Array2, Array4, Axis};
use opencv::core::{copy_make_border, flip, Point2f, Scalar, Size, Vec3b, Vector, BORDER_CONSTANT};
use opencv::imgproc::{resize, INTER_LINEAR};
use opencv::{imgproc, prelude::*};
use std::f64::consts::PI;
use std::f64::EPSILON;
use std::ops::Add;
use opencv::imgcodecs::imwrite;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeMode {
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

/// Convert a normalized bounding box into a ROI with optional scaling and rotation.
/// This function combines parts of DetectionsToRect and RectTransformation MediaPipe nodes.
pub fn bbox_to_roi(
    bbox: BBox, image_size: (i32, i32), rotation_keypoints: Option<Vec<(f64, f64)>>, scale: Option<(f64, f64)>,
    mode: Option<SizeMode>,
) -> Result<Rect, Error> {
    let scale_factor = scale.unwrap_or((1., 1.));
    let size_mode = mode.unwrap_or(SizeMode::Default);

    if !bbox.normalized() {
        return Err(Error::msg("bbox must be normalized"));
    }

    let (mut width, mut height) = select_roi_size(&bbox, image_size, size_mode)?;
    let (scale_x, scale_y) = scale_factor;

    (width, height) = (width * scale_x, height * scale_y);
    let cx = bbox.xmin + bbox.width() / 2.0;
    let cy = bbox.ymin + bbox.height() / 2.0;

    let rotation = if let Some(keypoints) = rotation_keypoints {
        if keypoints.len() < 2 {
            0.0
        } else {
            let (x0, y0) = keypoints[0];
            let (x1, y1) = keypoints[1];
            let angle = -(y0 - y1).atan2(x1 - x0);
            let two_pi = 2.0 * PI;
            let rotation = angle - two_pi * ((angle + PI) / two_pi).floor();
            rotation
        }
    } else {
        0.0
    };

    Ok(Rect {
        x_center: cx,
        y_center: cy,
        width,
        height,
        rotation,
        normalized: true,
    })
}

fn select_roi_size(bbox: &BBox, image_size: (i32, i32), size_mode: SizeMode) -> Result<(f64, f64), Error> {
    let abs_box = bbox.absolute(image_size);
    let (mut width, mut height) = (abs_box.width(), abs_box.height());
    let image_width = image_size.0 as f64;
    let image_height = image_size.1 as f64;

    let (width, height) = match size_mode {
        SizeMode::SquareLong => {
            let long_size = width.max(height);

            (width, height) = (long_size / image_width, long_size / image_height);
            (width, height)
        }
        SizeMode::SquareShort => {
            let short_side = width.min(height);
            (width, height) = (short_side / image_width, short_side / image_height);
            (width, height)
        }
        SizeMode::Default => (width, height),
    };

    Ok((width, height))
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
        .map(|detection| adjust_data(&detection, left as f32, top as f32, h_scale as f32, v_scale as f32))
        .collect()
}

/// Return the bounding box that encloses all landmarks in a given list.
/// This function combines the MediaPipe nodes LandmarksToDetectionCalculator and DetectionToRectCalculator.
pub fn bbox_from_landmarks(landmarks: &[Landmark]) -> Result<BBox, Error> {
    if landmarks.len() < 2 {
        return Err(Error::msg("landmarks must contain at least 2 items"));
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

    Ok(BBox { xmin, ymin, xmax, ymax })
}

/// Load an image into an array and return data, image size, and padding.
/// This function combines the mediapipe calculator-nodes ImageToTensor,
/// ImageCropping, and ImageTransformation into one function.
/// * Args:
///     - image (`Mat``): Input image; preferably RGB
///     - roi (`Option<Rect>`): Location within the image where to convert; can be `None`,
///             in which case the entire image is converted. Rotation is supported.
///     - output_size (`Option<(i32, i32)>`): Tuple of `(width, height)` describing the
///             output tensor size; defaults to ROI if `None`.
///     - keep_aspect_ratio (bool): `False` (default) will scale the image to
///             the output size; `True` will keep the ROI aspect ratio and apply
///             letterboxing.
///     - output_range (`(f64, f64)`): Tuple of `(min_val, max_val)` containing the
///             minimum and maximum value of the output tensor.
///             Defaults to (0, 1).
///     - flip_horizontal (`bool`): Flip the resulting image horizontally if set
///             to `True`. Default: `False`
///
/// * Returns:
///         (`ImageTensor`): Tensor data, padding for reversing letterboxing and
///         original image dimensions.
pub fn image_to_tensor(
    image: &Mat, roi: Option<Rect>, output_size: Option<(i32, i32)>, keep_aspect_ratio: bool, output_range: (f64, f64),
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
                top,
                bottom,
                left,
                right,
                BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;

            let mut resized_image = Mat::default();
            resize(&padded_image, &mut resized_image, Size::new(new_width, new_height), 0.0, 0.0, INTER_LINEAR)?;
            roi_image = resized_image;
        }

        let mut resized_image = Mat::default();
        resize(&roi_image, &mut resized_image, Size::new(output_size.0, output_size.1), 0.0, 0.0, INTER_LINEAR)
            .unwrap();
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

    imwrite("./test.jpg", &roi_image, &Vector::default());

    let mut tensors = Array3::<f32>::zeros((img_shape.width as usize, img_shape.height as usize, 3usize));

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
    let flattened: Vec<f32> = vec.into_iter().flat_map(|(x, y)| vec![x, y]).collect();
    let rows = flattened.len() / 2;
    Array2::from_shape_vec((rows, 2), flattened).unwrap()
}

fn perspective_transform_coeff(src_points: &Vec<(f32, f32)>, dst_points: &Vec<(f32, f32)>) -> Result<[f32; 8], Error> {
    if src_points.len() != 4 || dst_points.len() != 4 {
        return Err(Error::msg("src_points and dst_points must contain exactly 4 points each."));
    }

    let mut matrix = Vec::with_capacity(8 * 8);
    let mut b = Vec::with_capacity(8);

    for ((x, y), (x2, y2)) in dst_points.iter().zip(src_points.iter()) {
        matrix.push(vec![*x, *y, 1.0, 0.0, 0.0, 0.0, -x2 * x, -x2 * y]);
        matrix.push(vec![0.0, 0.0, 0.0, *x, *y, 1.0, -y2 * x, -y2 * y]);
        b.push(*x2);
        b.push(*y2);
    }

    let flat_matrix: Vec<f32> = matrix.into_iter().flatten().collect();
    // Convert the flattened matrix to a 2D array (8x8) and B to a 1D array (8)
    let a = nalgebra::DMatrix::from_row_slice(8, 8, &flat_matrix);
    let b = nalgebra::DVector::from_column_slice(&b);

    let coeffs = a.lu().solve(&b);

    match coeffs {
        Some(solution) => {
            let mut result = [0.0; 8];
            for i in 0..8 {
                result[i] = solution[i];
            }
            Ok(result)
        }
        None => Err(Error::msg("Failed to solve the linear system.")),
    }
}

pub fn project_landmarks(
    data: Array4<f32>, tensor_size: (i32, i32), image_size: (i32, i32), padding: (f64, f64, f64, f64),
    roi: Option<Rect>, flip_horizontal: bool,
) -> Result<Vec<Landmark>, Error> {
    let shape = (data.len() / 3, 3);
    let mut points: Array2<f32> = Array2::from_shape_vec(shape, data.flatten().to_vec())?;

    let (width, height) = tensor_size;

    points
        .index_axis_mut(Axis(1), 0)
        .mapv_inplace(|v| v / width as f32);
    points
        .index_axis_mut(Axis(1), 1)
        .mapv_inplace(|v| v / height as f32);
    points
        .index_axis_mut(Axis(1), 2)
        .mapv_inplace(|v| v / width as f32);

    if flip_horizontal {
        points.slice_mut(s![.., 0]).mapv_inplace(|x| x * -1.0 + 1.0);
    }

    if padding != (0.0, 0.0, 0.0, 0.0) {
        let (left, top, right, bottom) = padding;
        let h_scale = 1.0 - (left + right);
        let v_scale = 1.0 - (top + bottom);
        points
            .index_axis_mut(Axis(1), 0)
            .mapv_inplace(|v| ((v as f64 - left) / h_scale) as f32);
        points
            .index_axis_mut(Axis(1), 1)
            .mapv_inplace(|v| ((v as f64 - top) / v_scale) as f32);
        points
            .index_axis_mut(Axis(1), 2)
            .mapv_inplace(|v| ((v as f64 - 0.) / h_scale) as f32);
    }

    if let Some(roi) = roi {
        let norm_roi = roi.scaled((image_size.0 as f64, image_size.1 as f64), true);
        let sin = norm_roi.rotation.sin();
        let cos = norm_roi.rotation.cos();
        let matrix = Array2::from(vec![[cos as f32, sin as f32, 0.], [-sin as f32, cos as f32, 0.], [1., 1., 1.]]);
        points.index_axis_mut(Axis(1), 0).mapv_inplace(|v| v - 0.5);
        points.index_axis_mut(Axis(1), 1).mapv_inplace(|v| v - 0.5);
        points.index_axis_mut(Axis(1), 2).mapv_inplace(|v| v - 0.0);

        let mut rotated_points = points.clone();
        rotated_points
            .index_axis_mut(Axis(1), 2)
            .mapv_inplace(|v| v * 0.0);
        let rotated = rotated_points.dot(&matrix);

        points.index_axis_mut(Axis(1), 0).mapv_inplace(|v| v * 0.);
        points.index_axis_mut(Axis(1), 1).mapv_inplace(|v| v * 0.);
        points.index_axis_mut(Axis(1), 2).mapv_inplace(|v| v * 1.);

        points = points.add(rotated);

        points
            .index_axis_mut(Axis(1), 0)
            .mapv_inplace(|v| (v as f64 * norm_roi.width + norm_roi.x_center) as f32);
        points
            .index_axis_mut(Axis(1), 1)
            .mapv_inplace(|v| (v as f64 * norm_roi.height + norm_roi.y_center) as f32);
        points
            .index_axis_mut(Axis(1), 2)
            .mapv_inplace(|v| (v as f64 * norm_roi.width + 0.) as f32);

        let landmark: Vec<Landmark> = points
            .outer_iter()
            .map(|p| Landmark::new(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect();
        return Ok(landmark);
    }

    let landmark: Vec<Landmark> = points
        .outer_iter()
        .map(|p| Landmark::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    Ok(landmark)
}

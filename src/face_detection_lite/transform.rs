use ndarray::ArrayD;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::NdFloat;
use image::{DynamicImage, ImageBuffer, RgbImage, RgbaImage, GenericImageView, open};
use std::path::Path;
use anyhow::Error;
use ndarray::{Array3, Array};
use opencv::{core, imgproc, prelude::*};
use crate::face_detection_lite::types::{BBox, ImageTensor, Landmark, Rect};

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


fn sigmoid<T>(data: &ArrayBase<T, Dim<[usize; 1]>>) -> ArrayD<T>
    where
        T: NdFloat,
{
    1.0 / (1.0 + (-data).mapv(|x| x.exp()))
}



fn normalize_image(image: DynamicImage) -> DynamicImage {
    match image {
        DynamicImage::ImageRgb8(_) => image,
        _ => image.to_rgb8().into(),
    }
}

fn normalize_image_from_path(image_path: &str) -> DynamicImage {
    let image = open(&Path::new(image_path)).expect("Failed to open image");
    normalize_image(image)
}

fn normalize_image_from_array(array: &[u8], width: u32, height: u32) -> DynamicImage {
    let image = RgbImage::from_raw(width, height, array.to_vec())
        .expect("Failed to create image from array");
    DynamicImage::ImageRgb8(image)
}

pub fn bbox_from_landmarks(landmarks: &[Landmark]) -> Result<BBox, Error> {
    if landmarks.len() < 2 {
        return Err(Error::msg("landmarks must contain at least 2 items"))
    }

    let mut xmin = f32::INFINITY;
    let mut ymin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymax = f32::NEG_INFINITY;

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



// fn image_to_tensor(
//     image: DynamicImage,
//     roi: Option<Rect>,
//     output_size: Option<(u32, u32)>,
//     keep_aspect_ratio: bool,
//     output_range: (f32, f32),
//     flip_horizontal: bool,
// ) -> ImageTensor {
//     let img = image.to_rgb8();
//     let (img_width, img_height) = img.dimensions();
//
//     let roi = roi.unwrap_or_else(|| Rect {
//         x_center: 0.5,
//         y_center: 0.5,
//         width: 1.0,
//         height: 1.0,
//         rotation: 0.0,
//         normalized: true,
//     });
//
//     let roi = roi.scaled((img_width as f32, img_height as f32));
//     let output_size = output_size.unwrap_or((roi.width as f32, roi.height as f32));
//
//     let (width, height) = if keep_aspect_ratio {
//         roi.size()
//     } else {
//         output_size
//     };
//
//     // Transformation logic
//     let dst_points = vec![
//         (0., 0.), (width as f32, 0.), (width as f32, height as f32), (0., height as f32)
//     ];
//     let coeffs = perspective_transform_coeff(roi.points(), dst_points);
//     let mut roi_image = imgproc::warp_perspective(
//         &img, &coeffs, (width as i32, height as i32),
//         imgproc::INTER_LINEAR, core::BORDER_CONSTANT, core::Scalar::all(0)
//     ).unwrap();
//
//     let (mut pad_x, mut pad_y) = (0.0, 0.0);
//     if keep_aspect_ratio {
//         let out_aspect = output_size.1 as f32 / output_size.0 as f32;
//         let roi_aspect = roi.height / roi.width;
//         if out_aspect > roi_aspect {
//             pad_y = (1.0 - roi_aspect / out_aspect) / 2.0;
//         } else {
//             pad_x = (1.0 - out_aspect / roi_aspect) / 2.0;
//         }
//     }
//
//     if flip_horizontal {
//         roi_image = roi_image.fliph();
//     }
//
//     // Apply output range transformation
//     let min_val = output_range.0;
//     let max_val = output_range.1;
//     let tensor_data: Array3<f32> = Array::from_shape_vec(
//         (height as usize, width as usize, 3),
//         roi_image.to_vec().into_iter().map(|p| p as f32 * (max_val - min_val) / 255.0 + min_val).collect()
//     ).unwrap();
//
//     ImageTensor {
//         tensor_data,
//         padding: (pad_x, pad_y, pad_x, pad_y),
//         original_size: (img_width, img_height),
//     }
// }

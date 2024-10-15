use anyhow::Error;
use ndarray::Array2;
use opencv::core::{Mat};
use opencv::imgcodecs::{imdecode, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, COLOR_BGR2RGB};
use ndarray_linalg::{Scalar};

pub fn convert_image_to_mat(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = Mat::from_slice(im_bytes)?;

    // Decode the image
    let brg_img = imdecode(&img_as_mat, IMREAD_COLOR)?;

    let mut rgb_img = Mat::default();
    cvt_color(&brg_img, &mut rgb_img, COLOR_BGR2RGB, 0)?;

    // let mut res = Mat::default();
    // cvt_color(&rgb_img, &mut res, COLOR_BGR2RGB, 0)?;
    Ok(rgb_img)
}

/// l2_norm calculates the l2 normalized
///
/// # Arguments
/// * `arr` - `Array2<f32>`
///
/// # Returns
/// * `Array2<f32>`
pub fn l2_norm(arr: &Array2<f32>) -> Array2<f32> {
    let norm = arr.iter().map(|x| x.square()).sum::<f32>().sqrt();
    arr / norm
}


/// similarity_score calculates the cosine similarity
///
/// # Arguments
/// * `a` - &Vec<f32>
/// * `b` - &Vec<f32>
///
/// # Returns
/// * `f32`
pub fn similarity_score(a: &Vec<f32>, b: &Vec<f32>)  -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
    let norm_a = a.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|b| b.powi(2)).sum::<f32>().sqrt();
    let cosine = dot_product / (norm_a * norm_b);
    cosine
}
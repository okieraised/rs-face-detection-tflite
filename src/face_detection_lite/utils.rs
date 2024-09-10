use anyhow::Error;
use opencv::core::Mat;
use opencv::imgcodecs::{imdecode, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, COLOR_BGR2RGB};

pub fn convert_image_to_mat(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = Mat::from_slice(im_bytes)?;

    // Decode the image
    let img_as_arr_bgr = imdecode(&img_as_mat, IMREAD_COLOR)?;

    let mut img_as_arr_rgb = Mat::default();
    cvt_color(&img_as_arr_bgr, &mut img_as_arr_rgb, COLOR_BGR2RGB, 0)?;

    let mut res = Mat::default();
    cvt_color(&img_as_arr_rgb, &mut res, COLOR_BGR2RGB, 0)?;
    Ok(res)
}
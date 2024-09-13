use anyhow::Error;
use opencv::core::{Mat, MatTraitConst};
use opencv::imgcodecs::{imdecode, IMREAD_COLOR, IMREAD_UNCHANGED};
use opencv::imgproc::{cvt_color, COLOR_BGR2RGB, COLOR_GRAY2RGB, COLOR_RGBA2RGB};

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

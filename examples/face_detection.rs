use opencv::core::MatTraitConst;
use rs_face_detection_tfite::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel};
use rs_face_detection_tfite::face_detection_lite::face_landmark::face_detection_to_roi;
use rs_face_detection_tfite::face_detection_lite::utils::convert_image_to_mat;

fn main() {
    let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();

    // test data
    let im_bytes: &[u8] = include_bytes!("../test_data/man.jpg");
    let image = convert_image_to_mat(im_bytes).unwrap();
    let img_shape = image.size().unwrap();

    // Face detection
    let faces = face_detection.infer(&image, None).unwrap();
    let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();
    println!("face_roi: {:?}", face_roi)
}
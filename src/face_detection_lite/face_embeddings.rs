use std::path::PathBuf;
use anyhow::Error;
use ndarray::Array2;
use opencv::core::{Mat, MatTraitConst};
use tflite::FlatBufferModel;
use crate::face_detection_lite::face_landmark::{face_detection_to_roi, FaceLandmark};
use crate::face_detection_lite::transform::image_to_tensor;
use crate::face_detection_lite::types::{Detection, Landmark, Rect};

enum FeatureCount {
    Feature128,
    Feature512,
}
const MODEL_NAME_512: &str = "face_embeddings_512.tflite";

const MODEL_NAME_128: &str = "face_embeddings_128.tflite";

const IMG_SIZE: i32 = 160;

pub struct FaceEmbeddings {
    model_path: PathBuf,
    model: FlatBufferModel,
}


impl FaceEmbeddings {
    /// `FaceEmbeddings` extracts facial features as an array of 128 or 512 f32 elements
    pub fn new(model_path: Option<String>) -> Result<FaceEmbeddings, Error> {
        let mut model_path_buf: PathBuf;

        if let Some(path) = model_path {
            model_path_buf = PathBuf::from(path);
        } else {
            model_path_buf = PathBuf::from("./models/face_embeddings_512.tflite");
        }
        let model = FlatBufferModel::build_from_file(model_path_buf.clone())?;

        Ok(FaceEmbeddings {
            model_path: model_path_buf,
            model,
        })
    }

    pub fn infer(&self, image: &Mat, rect: Option<Rect>) -> Result<(), Error> {

        let img_shape = image.size()?;

        // let face_roi = face_detection_to_roi(detection, (img_shape.width, img_shape.height)).unwrap();
        let image_data = image_to_tensor(&image, rect, Some((IMG_SIZE, IMG_SIZE)), false, (0.0, 1.0), false)?;


        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel};
    use crate::face_detection_lite::utils::convert_image_to_mat;
    use opencv::core::MatTraitConst;
    use crate::face_detection_lite::face_landmark::face_detection_to_roi;

    #[test]
    fn test_face_embeddings_512() {
        let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();

        // let im_bytes: &[u8] = include_bytes!("../../test_data/man.jpg");
        let im_bytes: &[u8] = include_bytes!("/home/tripg/Documents/face/nv.png");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let img_shape = image.size().unwrap();

        let faces = face_detection.infer(&image, None).unwrap();
        println!("faces: {:?}", faces);

        let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height)).unwrap();

        let face_embeddings = FaceEmbeddings::new(None).unwrap();
        let lmks = face_embeddings.infer(&image, Some(face_roi)).unwrap();


    }
}
pub mod face_detection_lite;

#[cfg(test)]
mod tests {
    use crate::face_detection_lite::face_detection::{FaceDetection, FaceDetectionModel};
    use crate::face_detection_lite::face_landmark::{
        face_detection_to_roi, face_landmarks_to_render_data, FaceLandmark, FACE_LANDMARK_CONNECTIONS,
    };
    use crate::face_detection_lite::iris_landmark::{
        eye_landmarks_to_render_data, iris_landmarks_to_render_data, iris_roi_from_face_landmarks, IrisLandmark,
    };
    use crate::face_detection_lite::render::{detections_to_render_data, landmarks_to_render_data, render_to_image, Colors, Annotation};
    use crate::face_detection_lite::utils::convert_image_to_mat;
    use image::ImageReader;
    use opencv::core::MatTraitConst;
    use crate::face_detection_lite::types::Landmark;

    #[test]
    fn test_face_landmark() {
        let face_detection = FaceDetection::new(FaceDetectionModel::BackCamera, None).unwrap();

        // test data
        let im_bytes: &[u8] = include_bytes!("../test_data/man.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let img_shape = image.size().unwrap();

        // Face detection
        let faces = face_detection.infer(&image, None).unwrap();
        let face_roi = face_detection_to_roi(faces[0].clone(), (img_shape.width, img_shape.height), None).unwrap();

        // Face landmark
        let face_landmark = FaceLandmark::new(None).unwrap();
        let lmks = face_landmark.infer(&image, Some(face_roi)).unwrap();

        // Face irises
        let (left_eye_roi, right_eye_roi) = iris_roi_from_face_landmarks(lmks.clone(), (img_shape.width, img_shape.height)).unwrap();
        let iris_landmark = IrisLandmark::new(None).unwrap();

        let right_iris_lmk = iris_landmark.infer(&image, Some(right_eye_roi), Some(true)).unwrap();
        let left_iris_lmk = iris_landmark.infer(&image, Some(left_eye_roi), Some(false)).unwrap();

        // Draw face bounding box
        let render_data = detections_to_render_data(
            faces,
            Some(Colors::GREEN),
            None,
            4,
            2,
            true,
            None,
        );

        let img = ImageReader::open("./test_data/man.jpg")
            .unwrap()
            .decode()
            .unwrap();

        let res = render_to_image(&render_data, &img, None);
        res.save("./assets/man_bbox.png").unwrap();

        // Draw face landmarks
        let annotations = face_landmarks_to_render_data(lmks.clone(), Colors::RED, Colors::RED, Some(2.0), None);
        let res = render_to_image(&annotations, &img, None);
        res.save("./assets/man_landmark.png").unwrap();


        let right_iris_lmk = eye_landmarks_to_render_data(
            right_iris_lmk.eyeball_contour(),
            Colors::RED, Colors::RED, Some(2.0), None,
        );

        let left_iris_lmk = eye_landmarks_to_render_data(
            left_iris_lmk.eyeball_contour(),
            Colors::RED, Colors::RED, Some(2.0), None,
        );

        let mut iris_lmks: Vec<Annotation> = Vec::new();
        iris_lmks.extend(right_iris_lmk);
        iris_lmks.extend(left_iris_lmk);


        let res = render_to_image(&iris_lmks, &img, None);
        res.save("./assets/man_iris.png").unwrap();
    }
}

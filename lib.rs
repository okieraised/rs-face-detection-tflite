mod face_detection_lite;
mod models;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayD, IxDyn};

    #[test]
    fn it_works() {
        let data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let shape = (data.len() / 2, 2);
        let reshaped_data: Array2<f32> = Array2::from_shape_vec(shape, data).unwrap();
        println!("reshaped_data: {:?}", reshaped_data);
    }
}

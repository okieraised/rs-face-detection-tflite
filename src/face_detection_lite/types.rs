use ndarray::{s, Array, Array1, Array2, ArrayD, IxDyn, Zip};
use std::ops::Mul;
use opencv::core::Mat;

#[derive(Debug, Clone)]
pub struct ImageTensor {
    /// Tensor data obtained from an image with optional letterboxing.
    /// The data may contain an extra dimension for batching (the default).
    pub tensor_data: ArrayD<f32>, // ArrayD represents a dynamic-dimensional array in Rust.
    pub padding: (f64, f64, f64, f64), // Tuple for padding (left, top, right, bottom).
    pub original_size: (i32, i32),     // Tuple for the original image size (width, height).
}

impl ImageTensor {
    pub fn new(tensor_data: ArrayD<f32>, padding: (f64, f64, f64, f64), original_size: (i32, i32)) -> Self {
        Self {
            tensor_data,
            padding,
            original_size,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// The center of the rectangle (x, y).
    pub x_center: f64,
    pub y_center: f64,
    /// The width and height of the rectangle.
    pub width: f64,
    pub height: f64,
    /// The rotation of the rectangle in radians (clockwise).
    pub rotation: f64,
    /// Indicates whether properties are relative to image size (normalized).
    pub normalized: bool,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x_center: f64, y_center: f64, width: f64, height: f64, rotation: f64, normalized: bool) -> Self {
        Self {
            x_center,
            y_center,
            width,
            height,
            rotation,
            normalized,
        }
    }

    /// Return the size of the rectangle as a tuple (width, height).
    pub fn size(&self) -> (f64, f64) {
        let (w, h) = (self.width, self.height);
        if self.normalized {
            (w, h)
        } else {
            (w as i32 as f64, h as i32 as f64)
        }
    }

    /// Return a scaled version of the rectangle.
    pub fn scaled(&self, size: (f64, f64), normalize: bool) -> Rect {
        if self.normalized == normalize {
            return *self;
        }

        let (sx, sy) = if normalize { (1.0 / size.0, 1.0 / size.1) } else { size };

        Rect {
            x_center: self.x_center * sx,
            y_center: self.y_center * sy,
            width: self.width * sx,
            height: self.height * sy,
            rotation: self.rotation,
            normalized: normalize,
        }
    }

    /// Return the corners of the rectangle as a list of tuples `[(x, y), ...]`.
    pub fn points(&self) -> Vec<(f64, f64)> {
        let (x, y) = (self.x_center, self.y_center);
        let (w, h) = (self.width / 2.0, self.height / 2.0);
        let mut pts = vec![(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)];

        if self.rotation != 0.0 {
            let s = self.rotation.sin();
            let c = self.rotation.cos();
            let rotation_matrix = |pt: (f64, f64)| -> (f64, f64) {
                let (dx, dy) = (pt.0 - x, pt.1 - y);
                (x + dx * c - dy * s, y + dx * s + dy * c)
            };
            pts = pts.into_iter().map(rotation_matrix).collect();
        }

        pts
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
}

impl BBox {
    /// Create a new BBox
    pub fn new(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Self {
        Self { xmin, ymin, xmax, ymax }
    }

    /// Return the box as a tuple (xmin, ymin, xmax, ymax)
    pub fn as_tuple(&self) -> (f64, f64, f64, f64) {
        (self.xmin, self.ymin, self.xmax, self.ymax)
    }

    /// Calculate the width of the bounding box
    pub fn width(&self) -> f64 {
        self.xmax - self.xmin
    }

    /// Calculate the height of the bounding box
    pub fn height(&self) -> f64 {
        self.ymax - self.ymin
    }

    /// Check if the bounding box is empty (width or height is less than or equal to 0)
    pub fn empty(&self) -> bool {
        self.width() <= 0.0 || self.height() <= 0.0
    }

    /// Check if the bounding box coordinates are normalized (in range [0, 1])
    pub fn normalized(&self) -> bool {
        self.xmin >= -1.0 && self.xmax < 2.0 && self.ymin >= -1.0
    }

    /// Calculate the area of the bounding box
    pub fn area(&self) -> f64 {
        if self.empty() {
            0.0
        } else {
            self.width() * self.height()
        }
    }

    /// Calculate the intersection of this bounding box with another one
    pub fn intersect(&self, other: &BBox) -> Option<BBox> {
        let xmin = self.xmin.max(other.xmin);
        let ymin = self.ymin.max(other.ymin);
        let xmax = self.xmax.min(other.xmax);
        let ymax = self.ymax.min(other.ymax);

        if xmin < xmax && ymin < ymax {
            Some(BBox::new(xmin, ymin, xmax, ymax))
        } else {
            None
        }
    }

    /// Scale the bounding box by the given size
    pub fn scale(&self, size: (f64, f64)) -> BBox {
        let (sx, sy) = size;
        BBox::new(self.xmin * sx, self.ymin * sy, self.xmax * sx, self.ymax * sy)
    }

    /// Return the bounding box in absolute coordinates (if normalized)
    pub fn absolute(&self, size: (i32, i32)) -> BBox {
        if !self.normalized() {
            return *self;
        }
        self.scale((size.0 as f64, size.1 as f64))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Landmark {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub data: Array2<f32>,
    pub score: f32,
}

impl Detection {
    /// Create a new Detection
    pub fn new(data: Vec<f32>, score: f32) -> Self {
        assert!(data.len() >= 4, "Data must contain at least four elements for the bounding box");
        let shape = (data.len() / 2, 2);
        let reshaped_data: Array2<f32> = Array2::from_shape_vec(shape, data).unwrap();

        Self {
            data: reshaped_data,
            score,
        }
    }

    pub fn keypoint_count(&self) -> usize {
        self.data.nrows() - 2
    }

    // Get keypoint by index
    pub fn keypoint(&self, key: usize) -> (f32, f32) {
        let row = self.data.row(key + 2);
        (row[0], row[1])
    }

    // Get bounding box
    pub fn bbox(&self) -> BBox {
        let xmin = self.data[[0, 0]] as f64;
        let ymin = self.data[[0, 1]] as f64;
        let xmax = self.data[[1, 0]] as f64;
        let ymax = self.data[[1, 1]] as f64;
        BBox { xmin, ymin, xmax, ymax }
    }

    // Return a scaled version of the bounding box and keypoints
    pub fn scaled(&self, factor: f32) -> Detection {
        let scaled_data: Array2<f32> = self.data.mapv(|val| val * factor);

        Detection {
            data: scaled_data,
            score: self.score,
        }
    }

    pub fn scaled_by_image_size(&self, image_size: (i32, i32)) -> Detection {
        let mut result = self.data.clone();
        let scale = Array2::<f32>::from_shape_vec((1, 2), vec![image_size.0 as f32, image_size.1 as f32]).unwrap();
        let result = result.mul(scale);
        Detection {
            data: result,
            score: self.score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45];
        let detection = Detection::new(data, 0.85);

        println!("Detection: {:?}", detection);
        println!("Number of keypoints: {}", detection.keypoint_count());

        if let keypoint = detection.keypoint(0) {
            println!("First keypoint: {:?}", keypoint);
        }

        println!("Bounding box: {:?}", detection.bbox());

        let scaled_detection = detection.scaled(2.0);
        println!("Scaled Detection: {:?}", scaled_detection);
    }
}

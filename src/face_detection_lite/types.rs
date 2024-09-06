use ndarray::ArrayD;
use std::slice::Iter;

#[derive(Debug, Clone)]
pub struct ImageTensor {
    /// Tensor data obtained from an image with optional letterboxing.
    /// The data may contain an extra dimension for batching (the default).
    pub tensor_data: ArrayD<f32>, // ArrayD represents a dynamic-dimensional array in Rust.
    pub padding: (f32, f32, f32, f32), // Tuple for padding (left, top, right, bottom).
    pub original_size: (i32, i32), // Tuple for the original image size (width, height).
}

impl ImageTensor {
    pub fn new(tensor_data: ArrayD<f32>, padding: (f32, f32, f32, f32), original_size: (i32, i32)) -> Self {
        Self {
            tensor_data,
            padding,
            original_size,
        }
    }
}


use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// The center of the rectangle (x, y).
    pub x_center: f32,
    pub y_center: f32,
    /// The width and height of the rectangle.
    pub width: f32,
    pub height: f32,
    /// The rotation of the rectangle in radians (clockwise).
    pub rotation: f32,
    /// Indicates whether properties are relative to image size (normalized).
    pub normalized: bool,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x_center: f32, y_center: f32, width: f32, height: f32, rotation: f32, normalized: bool) -> Self {
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
    pub fn size(&self) -> (f32, f32) {
        let (w, h) = (self.width, self.height);
        if self.normalized {
            (w, h)
        } else {
            (w as i32 as f32, h as i32 as f32)
        }
    }

    /// Return a scaled version of the rectangle.
    pub fn scaled(&self, size: (f32, f32), normalize: bool) -> Rect {
        if self.normalized == normalize {
            return *self;
        }

        let (sx, sy) = if normalize {
            (1.0 / size.0, 1.0 / size.1)
        } else {
            size
        };

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
    pub fn points(&self) -> Vec<(f32, f32)> {
        let (x, y) = (self.x_center, self.y_center);
        let (w, h) = (self.width / 2.0, self.height / 2.0);
        let mut pts = vec![
            (x - w, y - h),
            (x + w, y - h),
            (x + w, y + h),
            (x - w, y + h),
        ];

        if self.rotation != 0.0 {
            let s = self.rotation.sin();
            let c = self.rotation.cos();
            let rotation_matrix = |pt: (f32, f32)| -> (f32, f32) {
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
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

impl BBox {
    /// Create a new BBox
    pub fn new(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Return the box as a tuple (xmin, ymin, xmax, ymax)
    pub fn as_tuple(&self) -> (f32, f32, f32, f32) {
        (self.xmin, self.ymin, self.xmax, self.ymax)
    }

    /// Calculate the width of the bounding box
    pub fn width(&self) -> f32 {
        self.xmax - self.xmin
    }

    /// Calculate the height of the bounding box
    pub fn height(&self) -> f32 {
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
    pub fn area(&self) -> f32 {
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
    pub fn scale(&self, size: (f32, f32)) -> BBox {
        let (sx, sy) = size;
        BBox::new(self.xmin * sx, self.ymin * sy, self.xmax * sx, self.ymax * sy)
    }

    /// Return the bounding box in absolute coordinates (if normalized)
    pub fn absolute(&self, size: (i32, i32)) -> BBox {
        if !self.normalized() {
            return *self;
        }
        self.scale((size.0 as f32, size.1 as f32))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Landmark {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}




#[derive(Debug, Clone)]
pub struct Detection {
    pub data: Vec<(f32, f32)>,  // Storing the data as a vector of tuples for (x, y) coordinates
    pub score: f32,             // Confidence score of the detection
}

impl Detection {
    /// Create a new Detection
    pub fn new(data: Vec<f32>, score: f32) -> Self {
        assert!(data.len() >= 4, "Data must contain at least four elements for the bounding box");

        // Reshape the data into a vector of (x, y) pairs
        let mut reshaped_data = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks(2) {
            reshaped_data.push((chunk[0], chunk[1]));
        }

        Self {
            data: reshaped_data,
            score,
        }
    }

    /// Get the number of keypoints (excluding the bounding box)
    pub fn keypoint_count(&self) -> usize {
        self.data.len() - 2
    }

    /// Get a keypoint by index
    pub fn keypoint(&self, index: usize) -> Option<(f32, f32)> {
        if index + 2 < self.data.len() {
            Some(self.data[index + 2])
        } else {
            None
        }
    }

    /// Return an iterator over the keypoints
    pub fn keypoints(&self) -> Iter<'_, (f32, f32)> {
        self.data[2..].iter()
    }

    /// Get the bounding box of the detection
    pub fn bbox(&self) -> BBox {
        let (xmin, ymin) = self.data[0];
        let (xmax, ymax) = self.data[1];
        BBox::new(xmin, ymin, xmax, ymax)
    }

    /// Return a scaled version of the detection
    pub fn scaled(&self, factor: f32) -> Detection {
        let scaled_data: Vec<(f32, f32)> = self.data.iter()
            .map(|&(x, y)| (x * factor, y * factor))
            .collect();

        Detection::new(scaled_data.into_iter().flat_map(|(x, y)| vec![x, y]).collect(), self.score)
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

        if let Some(keypoint) = detection.keypoint(0) {
            println!("First keypoint: {:?}", keypoint);
        }

        println!("Bounding box: {:?}", detection.bbox());

        let scaled_detection = detection.scaled(2.0);
        println!("Scaled Detection: {:?}", scaled_detection);
    }
}

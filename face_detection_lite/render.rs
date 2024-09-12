use crate::face_detection_lite::types::{Detection, Landmark};
use anyhow::Error;
use image::{DynamicImage, GenericImageView};
use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut};

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: Option<i32>,
}

impl Color {
    pub fn new(r: Option<i32>, g: Option<i32>, b: Option<i32>, a: Option<i32>) -> Self {
        Self {
            r: r.unwrap_or(0),
            g: g.unwrap_or(0),
            b: b.unwrap_or(0),
            a,
        }
    }
    pub fn as_tuple(&self) -> (i32, i32, i32, Option<i32>) {
        (self.r, self.g, self.b, self.a)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Colors;

impl Colors {
    pub const BLACK: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: None,
    };
    pub const RED: Color = Color {
        r: 255,
        g: 0,
        b: 0,
        a: None,
    };
    pub const GREEN: Color = Color {
        r: 0,
        g: 255,
        b: 0,
        a: None,
    };
    pub const BLUE: Color = Color {
        r: 0,
        g: 0,
        b: 255,
        a: None,
    };
    pub const PINK: Color = Color {
        r: 255,
        g: 0,
        b: 255,
        a: None,
    };
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: None,
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn as_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    pub fn scaled(&self, factor: (f64, f64)) -> Self {
        let (sx, sy) = factor;
        Point {
            x: self.x * sx,
            y: self.y * sy,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RectOrOval {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
    oval: bool,
}

impl RectOrOval {
    pub fn new(left: f64, top: f64, right: f64, bottom: f64, oval: bool) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            oval,
        }
    }

    pub fn as_tuple(&self) -> (f64, f64, f64, f64) {
        (self.left, self.top, self.right, self.bottom)
    }

    pub fn scaled(&self, factor: (f64, f64)) -> Self {
        let (sx, sy) = factor;
        RectOrOval {
            left: self.left * sx,
            top: self.top * sy,
            right: self.right * sx,
            bottom: self.bottom * sy,
            oval: self.oval,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FilledRectOrOval {
    rect: RectOrOval,
    fill: Color,
}

impl FilledRectOrOval {
    pub fn new(rect: RectOrOval, fill: Color) -> Self {
        Self { rect, fill }
    }

    pub fn scaled(&self, factor: (f64, f64)) -> Self {
        FilledRectOrOval {
            rect: self.rect.scaled(factor),
            fill: self.fill, // Color remains the same, as it doesn't scale
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Line {
    x_start: f64,
    y_start: f64,
    x_end: f64,
    y_end: f64,
    dashed: bool,
}

impl Line {
    pub fn new(x_start: f64, y_start: f64, x_end: f64, y_end: f64, dashed: bool) -> Self {
        Self {
            x_start,
            y_start,
            x_end,
            y_end,
            dashed,
        }
    }

    pub fn as_tuple(&self) -> (f64, f64, f64, f64) {
        (self.x_start, self.y_start, self.x_end, self.y_end)
    }

    // Method to return a scaled line
    pub fn scaled(&self, factor: (f64, f64)) -> Self {
        let (sx, sy) = factor;
        Line {
            x_start: self.x_start * sx,
            y_start: self.y_start * sy,
            x_end: self.x_end * sx,
            y_end: self.y_end * sy,
            dashed: self.dashed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AnnotationData {
    Point(Point),
    RectOrOval(RectOrOval),
    FilledRectOrOval(FilledRectOrOval),
    Line(Line),
}

impl AnnotationData {
    pub fn scaled(&self, factor: (f64, f64)) -> Self {
        match self {
            AnnotationData::Point(point) => AnnotationData::Point(point.scaled(factor)),
            AnnotationData::RectOrOval(rect) => AnnotationData::RectOrOval(rect.scaled(factor)),
            AnnotationData::FilledRectOrOval(filled_rect) => {
                AnnotationData::FilledRectOrOval(filled_rect.scaled(factor))
            }
            AnnotationData::Line(line) => AnnotationData::Line(line.scaled(factor)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Annotation {
    data: Vec<AnnotationData>,
    normalized_positions: bool,
    thickness: f64,
    color: Color,
}

impl Annotation {
    // Constructor for Annotation
    pub fn new(data: Vec<AnnotationData>, normalized_positions: bool, thickness: f64, color: Color) -> Self {
        Self {
            data,
            normalized_positions,
            thickness,
            color,
        }
    }

    pub fn scaled(&self, factor: (f64, f64)) -> Result<Self, Error> {
        if !self.normalized_positions {
            return Err(Error::msg("position data must be normalized"));
        }

        let scaled_data = self
            .data
            .iter()
            .map(|item| item.scaled(factor))
            .collect::<Vec<AnnotationData>>();

        Ok(Annotation {
            data: scaled_data,
            normalized_positions: false,
            thickness: self.thickness,
            color: self.color,
        })
    }
}

/// `detections_to_render_data` converts detections to render data.
/// This is an implementation of the MediaPipe DetectionToRenderDataCalculator node with keypoints added.
/// * Args:
///
///     - detections (`Vec<Detection>`): List of detects, which will be converted to individual points and a bounding box rectangle for rendering.
///     - bounds_color (`Option<Color>`): Color of the bounding box; if `None` the bounds won't be rendered.
///     - keypoint_color (`Option<Color>`): Color of the keypoints that will be rendered as points; set to `None` to disable keypoint rendering.
///     - line_width (`i32`): Thickness of the lines in viewport units (e.g. pixels).
///     - point_width (`i32`): Size of the keypoints in viewport units (e.g. pixels).
///     - normalized_positions (`bool`): Flag indicating whether the detections contain normalised data (e.g. range [0,1]).
///     - output (`Option<Vec<Annotation>>`): Optional render data instance to add the items to.
///       If not provided, a new instance of `RenderData` will be created.
///       Use this to add multiple landmark detections into a single render data bundle.
///
/// * Returns:
///    `Vec<Annotation>` - Vector of annotations for rendering landmarks.
pub fn detections_to_render_data(
    detections: Vec<Detection>, bounds_color: Option<Color>, keypoint_color: Option<Color>, line_width: i32,
    point_width: i32, normalized_positions: bool, output: Option<Vec<Annotation>>,
) -> Vec<Annotation> {
    fn to_rect(detection: &Detection) -> RectOrOval {
        let bbox = &detection.bbox();
        RectOrOval::new(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, false)
    }

    let mut annotations: Vec<Annotation> = vec![];
    if let Some(bounds_color) = bounds_color {
        if line_width > 0 {
            let bounds_data = detections
                .iter()
                .map(|detection| AnnotationData::RectOrOval(to_rect(detection)))
                .collect::<Vec<_>>();

            let bounds_annotation = Annotation::new(bounds_data, normalized_positions, line_width as f64, bounds_color);
            annotations.push(bounds_annotation);
        }
    }

    if let Some(keypoint_color) = keypoint_color {
        if point_width > 0 {
            let points = Annotation {
                data: detections
                    .iter()
                    .flat_map(|detection| {
                        detection
                            .data
                            .rows()
                            .into_iter()
                            .map(|row| {
                                let (x, y) = (row[0] as f64, row[1] as f64);
                                AnnotationData::Point(Point { x, y })
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect(),
                normalized_positions,
                thickness: point_width as f64,
                color: keypoint_color,
            };
            annotations.push(points);
        }
    }

    let mut output = output.unwrap_or_else(Vec::new);
    output.extend(annotations);

    output
}

pub fn landmarks_to_render_data(
    landmarks: Vec<Landmark>, landmark_connections: Vec<(i32, i32)>, landmark_color: Option<Color>,
    connection_color: Option<Color>, thickness: Option<f32>, normalized_positions: Option<bool>,
    output: Option<Vec<Annotation>>,
) -> Vec<Annotation> {
    let landmark_color = landmark_color.unwrap_or(Colors::RED);
    let connection_color = connection_color.unwrap_or(Colors::RED);
    let thickness = thickness.unwrap_or(1.) as f64;
    let normalized_positions = normalized_positions.unwrap_or(true);

    let lines: Vec<Line> = landmark_connections
        .iter()
        .map(|&(start, end)| {
            Line::new(
                landmarks[start as usize].x, landmarks[start as usize].y, landmarks[end as usize].x,
                landmarks[end as usize].y, false,
            )
        })
        .collect();

    let points: Vec<Point> = landmarks
        .iter()
        .map(|landmark| Point::new(landmark.x, landmark.y))
        .collect();

    let line_anno: Vec<AnnotationData> = lines
        .iter()
        .map(|&l| AnnotationData::Line(l))
        .collect::<Vec<_>>();
    let point_anno: Vec<AnnotationData> = points
        .iter()
        .map(|&l| AnnotationData::Point(l))
        .collect::<Vec<_>>();

    let line_annotation = Annotation::new(line_anno, normalized_positions, thickness, connection_color);
    let point_annotation = Annotation::new(point_anno, normalized_positions, thickness, landmark_color);

    if let Some(mut output_vec) = output {
        output_vec.push(line_annotation.clone());
        output_vec.push(point_annotation.clone());
        output_vec.clone()
    } else {
        vec![line_annotation, point_annotation]
    }
}

// pub fn render_to_image(annotations: Vec<Annotation>, image: &Mat, blend: Option<bool>) {
//     let blend_mode = blend.unwrap_or(false);
//
// }

// pub fn render_to_image(
//     annotations: &Vec<Annotation>,
//     image: &DynamicImage,
//     blend_mode: Option<bool>,
// ) -> DynamicImage {
//
//     let blend = blend_mode.unwrap_or(false);
//
//
//     let (width, height) = image.dimensions();
//     let mut img = image.to_rgba8();
//
//     for annotation in annotations {
//         let scaled = if annotation.normalized_positions {
//             let scale_x = width as f64;
//             let scale_y = height as f64;
//             let scaled_thickness = annotation.thickness;
//             let scaled_color = annotation.color.clone();
//             let scaled_data = annotation.data.iter()
//                 .map(|item| match item {
//                     AnnotationData::Point(p) => {
//                         AnnotationData::Point(Point { x: p.x * scale_x as f64, y: p.y * scale_y as f64 })
//                     },
//                     AnnotationData::RectOrOval(r) => {
//                         AnnotationData::RectOrOval(RectOrOval {
//                             left: r.left * scale_x as f64,
//                             top: r.top * scale_y as f64,
//                             right: r.right * scale_x as f64,
//                             bottom: r.bottom * scale_y as f64,
//                             oval: r.oval,
//                         })
//                     },
//                     AnnotationData::FilledRectOrOval(f) => {
//                         AnnotationData::FilledRectOrOval(FilledRectOrOval {
//                             rect: RectOrOval {
//                                 left: f.rect.left * scale_x,
//                                 top: f.rect.top * scale_y,
//                                 right: f.rect.right * scale_x,
//                                 bottom: f.rect.bottom * scale_y,
//                                 oval: f.rect.oval,
//                             },
//                             fill: f.fill.clone(),
//                         })
//                     },
//                     AnnotationData::Line(l) => {
//                         AnnotationData::Line(Line {
//                             x_start: l.x_start * scale_x,
//                             y_start: l.y_start * scale_y,
//                             x_end: l.x_end * scale_x,
//                             y_end: l.y_end * scale_y,
//                             dashed: l.dashed,
//                         })
//                     },
//                 })
//                 .collect::<Vec<_>>();
//
//             Annotation {
//                 data: scaled_data,
//                 normalized_positions: false,
//                 thickness: scaled_thickness,
//                 color: scaled_color,
//             }
//         } else {
//             annotation.clone()
//         };
//
//         let thickness = scaled.thickness as u32;
//         let color = scaled.color;
//
//         for item in scaled.data {
//             match item {
//                 AnnotationData::Point(p) => {
//                     let w = (thickness / 2).max(1);
//                     let x = p.x as u32;
//                     let y = p.y as u32;
//                     let rect = imageproc::rect::Rect::at((x - w) as i32, (y - w) as i32)
//                         .of_size(w * 2, w * 2);
//                     draw_filled_rect_mut(&mut img, rect, color, color);
//                 },
//                 AnnotationData::Line(l) => {
//                     let x_start = l.x_start as i32;
//                     let y_start = l.y_start as i32;
//                     let x_end = l.x_end as i32;
//                     let y_end = l.y_end as i32;
//                     draw_line_segment_mut(&mut img, (x_start, y_start), (x_end, y_end), color);
//                 },
//                 AnnotationData::RectOrOval(r) => {
//                     let rect = imageproc::rect::Rect::at(r.left as i32, r.top as i32)
//                         .of_size((r.right - r.left) as u32, (r.bottom - r.top) as u32);
//                     if r.oval {
//                         draw_filled_rect_mut(&mut img, rect, color);
//                     } else {
//                         draw_filled_rect_mut(&mut img, rect, color, color);
//                     }
//                 },
//                 AnnotationData::FilledRectOrOval(f) => {
//                     let rect = imageproc::rect::Rect::at(f.rect.left as i32, f.rect.top as i32)
//                         .of_size((f.rect.right - f.rect.left) as u32, (f.rect.bottom - f.rect.top) as u32);
//                     let fill_color = f.fill.to_rgba();
//                     if f.rect.oval {
//                         draw_filled_rect_mut(&mut img, rect, fill_color);
//                     } else {
//                         draw_filled_rect_mut(&mut img, rect, fill_color, color);
//                     }
//                 }
//             }
//         }
//     }
//
//     DynamicImage::ImageRgba8(img)
// }

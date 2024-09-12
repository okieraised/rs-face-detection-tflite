use crate::face_detection_lite::types::{BBox, Detection};
use ndarray::{Array, Array2, ArrayD, IxDyn, Zip};

/// Return intersection-over-union similarity of two bounding boxes
fn overlap_similarity(box1: &BBox, box2: &BBox) -> f64 {
    if let Some(intersection) = box1.intersect(box2) {
        let intersect_area = intersection.area();
        let denominator = box1.area() + box2.area() - intersect_area;
        if denominator > 0.0 {
            intersect_area / denominator
        } else {
            0.0
        }
    } else {
        0.0
    }
}

pub fn _non_maximum_suppression(
    indexed_scores: Vec<(usize, f32)>, detections: Vec<Detection>, min_suppression_threshold: f32,
    min_score: Option<f32>,
) -> Vec<Detection> {
    let mut kept_boxes: Vec<BBox> = Vec::new();
    let mut outputs: Vec<Detection> = Vec::new();

    for &(index, score) in &indexed_scores {
        // Exit loop if remaining scores are below threshold
        if let Some(min_score) = min_score {
            if score < min_score {
                break;
            }
        }

        let detection = &detections[index];
        let bbox = detection.bbox();
        let mut suppressed = false;

        for kept in &kept_boxes {
            let similarity = overlap_similarity(kept, &bbox);
            if similarity > min_suppression_threshold as f64 {
                suppressed = true;
                break;
            }
        }

        if !suppressed {
            outputs.push(detection.clone());
            kept_boxes.push(bbox);
        }
    }

    outputs
}

/// Return only most significant detections; merge similar detections
pub fn weighted_non_maximum_suppression(
    indexed_scores: Vec<(usize, f32)>, detections: Vec<Detection>, min_suppression_threshold: f32,
    min_score: Option<f32>,
) -> Vec<Detection> {
    let mut remaining_indexed_scores = indexed_scores;
    let mut remaining: Vec<(usize, f32)> = Vec::new();
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    let mut outputs: Vec<Detection> = Vec::new();

    while !remaining_indexed_scores.is_empty() {
        let detection = &detections[remaining_indexed_scores[0].0];

        // Exit loop if remaining scores are below threshold
        if let Some(min_score) = min_score {
            if detection.score < min_score {
                break;
            }
        }

        let num_prev_indexed_scores = remaining_indexed_scores.len();
        let detection_bbox = detection.bbox();

        remaining.clear();
        candidates.clear();

        let mut weighted_detection = detection.clone();

        for &(index, score) in &remaining_indexed_scores {
            let remaining_bbox = detections[index].bbox();
            let similarity = overlap_similarity(&remaining_bbox, &detection_bbox);

            if similarity > min_suppression_threshold as f64 {
                candidates.push((index, score));
            } else {
                remaining.push((index, score));
            }
        }

        if !candidates.is_empty() {
            let num_features = detection.data.nrows();
            let mut weighted = Array2::<f32>::zeros((num_features, 2));
            let mut total_score = 0.0;

            for &(index, score) in &candidates {
                total_score += score;
                Zip::from(&mut weighted)
                    .and(&detections[index].data)
                    .for_each(|w, &d| {
                        *w += d * score;
                    });
            }

            weighted /= total_score;
            let (w_det, _) = weighted.into_raw_vec_and_offset();

            weighted_detection = Detection::new(w_det, detection.score);
        }

        outputs.push(weighted_detection);

        // Exit the loop if the number of indexed scores didn't change
        if num_prev_indexed_scores == remaining.len() {
            break;
        }
        remaining_indexed_scores = remaining.clone();
    }

    outputs
}

/// Return only significant detections.
pub fn non_maximum_suppression(
    detections: Vec<Detection>, min_suppression_threshold: f32, min_score: Option<f32>, weighted: bool,
) -> Vec<Detection> {
    let mut scores: Vec<(usize, f32)> = detections
        .iter()
        .enumerate()
        .map(|(n, detection)| (n, detection.score))
        .collect();

    // Sort scores in descending order
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if weighted {
        weighted_non_maximum_suppression(scores, detections, min_suppression_threshold, min_score)
    } else {
        _non_maximum_suppression(scores, detections, min_suppression_threshold, min_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray() {
        let mut weighted: Array<f32, IxDyn> = ArrayD::<f32>::zeros(IxDyn(&[2 + 1, 2]));
        println!("weighted: {:?}", weighted);
    }
}

pub struct Color {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: Option<i32>
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

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum Colors {
//     /// Keep width and height as calculated.
//     Default = 0,
//
//     /// Make square using `max(width, height)`.
//     SquareLong = 1,
//
//     /// Make square using `min(width, height)`.
//     SquareShort = 2,
// }

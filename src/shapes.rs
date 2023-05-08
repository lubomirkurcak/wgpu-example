use cgmath::prelude::*;
use cgmath::Point2;
use cgmath::Vector2;
use crate::Vertex;
use crate::model::Mesh;
use crate::model::Model;

struct ShapeVertex {
    position: Point2<f32>,
    // normal: cgmath::Vector3<f32>,
}

impl Vertex for ShapeVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ShapeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct Triangles;

impl Triangles {
    // pub fn rectangle(min: Point2<f32>, max: Point2<f32>) -> Mesh {
    //     Mesh {
    //         name: "Rectangle".into(),
    //     }
    //     vec![
    //         ShapeVertex { position: min },
    //         ShapeVertex {
    //             position: Point2::new(max.x, min.y),
    //         },
    //         ShapeVertex { position: max },
    //         ShapeVertex { position: min },
    //         ShapeVertex { position: max },
    //         ShapeVertex {
    //             position: Point2::new(min.x, max.y),
    //         },
    //     ]
    // }

    // pub fn arrow(dir: cgmath::Vector2<f32>) -> Vec<ShapeVertex> {
    //     let p = dir.perp();
    //     let mut points = vec![
    //         cgmath::Point3::new(0.0, 0.0, 0.0),
    //         cgmath::Point3::new(0.0, 0.0, 1.0),
    //         cgmath::Point3::new(0.0, 0.0, 1.0),
    //         cgmath::Point3::new(0.0, 0.0, 1.0),
    //         cgmath::Point3::new(0.0, 0.0, 1.0),
    //         cgmath::Point3::new(0.0, 0.0, 1.0),
    //     ];
    //     points
    // }
}

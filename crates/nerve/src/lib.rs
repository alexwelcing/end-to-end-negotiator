//! Nerve in Rust: interfaces as spec-checked energy networks, with
//! Dragology-style drags, trained on the local GPU.

pub mod continuous;
pub mod drag;
pub mod encode;
pub mod exp;
pub mod expr;
pub mod gen;
pub mod jepa;
pub mod machine;
pub mod machine_api;
pub mod oracle;
pub mod picture;
pub mod render;
pub mod serve;
pub mod shader;
#[cfg(feature = "server")]
pub mod site;
#[cfg(feature = "server")]
pub mod smoke;
pub mod spec;
pub mod train;
pub mod value;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

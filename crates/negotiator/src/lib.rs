//! Rust + CUDA port of "Deal or No Deal? End-to-End Learning for Negotiation
//! Dialogues" (Lewis et al., 2017).

pub mod agent;
pub mod cli;
pub mod data;
pub mod device;
pub mod dialog;
pub mod domain;
pub mod engine;
pub mod metric;
pub mod model;
pub mod nn;
pub mod optim;

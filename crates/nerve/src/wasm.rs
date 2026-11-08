//! The lab in the browser: the same `App` that answers `nerve serve`, driven from JavaScript with
//! no server. Specs, machine documents and checkpoints are passed in as text and bytes; every API
//! path the pages call is answered by `Room::call`, so the pages run unchanged against either.

use crate::machine::MachineSpec;
use crate::machine_api::MachineEntry;
use crate::serve::{App, Entry};
use crate::spec::Spec;
use crate::train::load_learned_from_bytes;
use candle_core::Device;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Room {
    app: App,
}

fn js(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen]
impl Room {
    /// An empty room; `lambda` weighs the learned energy against the pixel gap in drags, as `nerve serve --lambda`.
    #[wasm_bindgen(constructor)]
    pub fn new(lambda: f64) -> Room {
        Room { app: App::new(Vec::new(), lambda) }
    }

    /// Adds an interface from its spec text and returns its name.
    pub fn add_spec(&mut self, text: &str) -> Result<String, JsError> {
        let spec = Spec::parse(text).map_err(js)?;
        let name = spec.name.clone();
        self.app.add_entry(Entry { spec, model: None, source: text.to_string() });
        Ok(name)
    }

    /// Loads a trained network for an interface already added: the `model.json` text and the
    /// `model.safetensors` bytes of a checkpoint, as `nerve train` writes them.
    pub fn add_model(&mut self, spec: &str, meta_json: &str, weights: &[u8]) -> Result<(), JsError> {
        let s = self.app.entry(spec).map_err(js)?.spec.clone();
        let model = load_learned_from_bytes(s, meta_json, weights, &Device::Cpu).map_err(js)?;
        self.app.set_model(spec, model).map_err(js)
    }

    /// Adds a machine from its document; its block types must already be added as specs.
    pub fn add_machine(&mut self, text: &str) -> Result<String, JsError> {
        let spec = {
            let app = &self.app;
            MachineSpec::parse(text, &|ty| Ok(app.entry(ty)?.spec.clone())).map_err(js)?
        };
        let name = spec.name.clone();
        self.app.add_machine(MachineEntry { spec, source: text.to_string() });
        Ok(name)
    }

    /// Whether an interface has a network loaded.
    pub fn has_model(&self, spec: &str) -> bool {
        self.app.entry(spec).ok().is_some_and(|e| e.model.is_some())
    }

    /// Answers an API path (`/api/specs`, `/api/<spec>/step`, `/api/machine/<name>/tick`, ...) with
    /// a JSON body, exactly as the HTTP server would; the answer is JSON text.
    pub fn call(&self, path: &str, body: &str) -> Result<String, JsError> {
        let body: serde_json::Value = if body.trim().is_empty() { serde_json::Value::Null } else { serde_json::from_str(body).map_err(|e| JsError::new(&e.to_string()))? };
        Ok(self.app.dispatch(path, &body).map_err(js)?.to_string())
    }
}

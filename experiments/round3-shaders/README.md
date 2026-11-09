# Round 3: the picture as a function of the pixel

`nerve train-shader` / `nerve shader-train --remote hf` (the Python twin on a Hugging Face a10g-small Job), 20,000 steps,
8 states × 4,096 random pixels per step, Adam 1e-3 cosine, training raster = 4× the scene's. Scored by the engine
(`nerve eval-shader`): mean absolute pixel error against the render oracle at the training raster, and at the
scene's own raster with the 4× picture box-filtered down (what a display does). The picture networks' numbers are
their committed `picture.json` evals at the scene's raster.

| spec | trained at | shader error at 4× | shader at scene raster (supersampled) | picture network at scene raster | params | steps | GPU time |
|---|---|---|---|---|---|---|---|
| bell | 768×768 | 0.039% | 0.176% | 1.94% | 241,027 | 20,000 | 117 s |
| catapult | 768×640 | 0.013% | 0.029% | 1.37% | 237,187 | 20,000 | 81 s |
| clock | 768×256 | 0.380% | 0.519% | 1.21% | 241,027 | 20,000 | 115 s |
| counter | 768×256 | 0.035% | 0.555% | 2.87% | 241,539 | 20,000 | 115 s |
| digit | 640×768 | 0.019% | 0.276% | 0.65% | 241,027 | 20,000 | 117 s |
| dominoes | 768×128 | 0.026% | 0.058% | 1.31% | 241,795 | 20,000 | 114 s |
| fifteen | 768×512 | 0.086% | (scoring) | 2.29% | 255,363 | 20,000 | 117 s |
| hanoi | 768×256 | 0.122% | 0.554% | 1.72% | 248,451 | 20,000 | 115 s |
| insert_remove | 768×512 | 0.085% | (scoring) | 2.61% | 250,499 | 20,000 | 118 s |
| kanban | 768×384 | 0.087% | (scoring) | 1.54% | 257,667 | 20,000 | 116 s |
| linked_sliders | 768×384 | 0.417% | (scoring) | 0.78% | 243,075 | 20,000 | 116 s |
| marble | 768×256 | 0.517% | 0.581% | 0.76% | 241,283 | 20,000 | 115 s |
| negotiate | 768×512 | 0.123% | (scoring) | 1.29% | 246,915 | 20,000 | 116 s |
| rail | 768×128 | 0.277% | 0.345% | 0.61% | 238,979 | 20,000 | 114 s |
| relay | 768×512 | 0.018% | (scoring) | 1.64% | 236,675 | 20,000 | 81 s |
| reorder | 768×128 | 0.043% | (scoring) | 1.98% | 254,083 | 20,000 | 113 s |
| rush_hour | 768×640 | 0.164% | (scoring) | 1.90% | 256,387 | 20,000 | 119 s |
| scheduler | 768×768 | 0.174% | (scoring) | 1.50% | 256,131 | 20,000 | 120 s |
| seesaw | 768×256 | 0.012% | (scoring) | 1.22% | 236,931 | 20,000 | 86 s |
| slider | 768×128 | 0.114% | 0.730% | 3.31% | 245,379 | 20,000 | 114 s |
| slider_continuous | 768×128 | 0.161% | (scoring) | 1.13% | 238,979 | 20,000 | 113 s |
| sokoban | 768×512 | 0.182% | (scoring) | 2.09% | 254,595 | 20,000 | 118 s |
| three_way_switch | 768×640 | 0.084% | (scoring) | 2.39% | 237,443 | 20,000 | 95 s |
| todo | 640×768 | 0.098% | (scoring) | 2.84% | 251,523 | 20,000 | 120 s |
| toggle | 768×384 | 0.028% | 0.231% | 1.50% | 236,675 | 20,000 | 86 s |
| tracks | 768×640 | 0.315% | (scoring) | 1.06% | 241,539 | 20,000 | 120 s |

The shader is one network evaluated once per pixel, so the lab draws it as a WebGL fragment shader at the display's
resolution (`crates/nerve/web/index.html`, "shader (GPU)"): 573×286 px in 0.1 ms on a laptop GPU, against 30–100 ms
for a 192-px picture network in WebAssembly. GPU output matches the engine's CPU draw to 0.001/255 per channel
(scripts in this session's record; `scripts/wasm_smoke.mjs` to be extended).

Point-sampling the shader at the scene's raster (no supersampling) is not a fair score: thin lines alias, exactly as
they do for any point sampler, and it reads 2–4% where the supersampled picture reads 0.3–0.6%. The engine reports
both (`mae_native`, `mae_native_point`).

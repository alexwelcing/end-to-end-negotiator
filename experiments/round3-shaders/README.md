# Round 3: the picture as a function of the pixel

`nerve train-shader` / `nerve shader-train --remote hf` (the Python twin on a Hugging Face a10g-small Job), 20,000 steps,
8 states × 4,096 random pixels per step, Adam 1e-3 cosine, training raster = 4× the scene's. Scored by the engine
(`nerve eval-shader`, 16 of each shader's training states): mean absolute pixel error against the render oracle at
the training raster; at the scene's own raster with the 4× picture box-filtered down (what a display does); and
**off the lattice**, drawn at 0.9× the training raster where no pixel centre coincides with a training sample (what
the display does at any other size, including full screen). The picture networks' numbers are
their committed `picture.json` evals at the scene's raster.

| spec | trained at | training | shader error at 4× | at scene raster (supersampled) | off the lattice (0.9×) | picture network at scene raster | params | steps | GPU time |
|---|---|---|---|---|---|---|---|---|---|
| bell | 768×768 | lattice | 0.039% | 0.176% | 0.079% | 1.939% | 241,027 | 20,000 | 117 s |
| catapult | 768×640 | lattice | 0.013% | 0.029% | 0.016% | 1.368% | 237,187 | 20,000 | 81 s |
| clock | 768×256 | wide | 0.007% | 0.164% | 0.157% | 1.211% | 478,275 | 40,000 | 427 s |
| counter | 768×256 | jittered | 0.067% | 0.548% | 0.350% | 2.868% | 240,515 | 20,000 | 114 s |
| digit | 640×768 | wide | 0.004% | 0.271% | 0.037% | 0.651% | 478,275 | 40,000 | 427 s |
| dominoes | 768×128 | jittered | 0.013% | 0.042% | 0.195% | 1.314% | 240,771 | 20,000 | 112 s |
| fifteen | 768×512 | lattice | 0.087% | 0.813% | 0.591% | 2.292% | 255,363 | 20,000 | 117 s |
| hanoi | 768×256 | jittered | 0.142% | 0.549% | 0.411% | 1.723% | 247,427 | 20,000 | 115 s |
| insert_remove | 768×512 | lattice | 0.083% | 0.637% | 0.281% | 2.606% | 250,499 | 20,000 | 118 s |
| kanban | 768×384 | lattice | 0.089% | 0.587% | 0.459% | 1.539% | 257,667 | 20,000 | 116 s |
| linked_sliders | 768×384 | jittered | 0.377% | 0.474% | 0.522% | 0.776% | 242,051 | 20,000 | 114 s |
| marble | 768×256 | wide | 0.009% | 0.107% | 0.089% | 0.764% | 478,531 | 40,000 | 424 s |
| negotiate | 768×512 | lattice | 0.122% | 0.705% | 0.442% | 1.294% | 246,915 | 20,000 | 116 s |
| rail | 768×128 | jittered | 0.166% | 0.246% | 0.172% | 0.605% | 237,955 | 20,000 | 112 s |
| relay | 768×512 | lattice | 0.018% | 0.293% | 0.114% | 1.641% | 236,675 | 20,000 | 81 s |
| reorder | 768×128 | jittered | 0.105% | 1.850% | 1.544% | 1.976% | 253,059 | 20,000 | 113 s |
| rush_hour | 768×640 | wide | 0.025% | 0.731% | 0.332% | 1.901% | 493,635 | 40,000 | 432 s |
| scheduler | 768×768 | wide | 0.011% | 0.400% | 0.215% | 1.502% | 493,379 | 40,000 | 438 s |
| seesaw | 768×256 | jittered | 0.021% | 0.110% | 0.047% | 1.225% | 235,907 | 20,000 | 86 s |
| slider | 768×128 | jittered | 0.094% | 0.701% | 0.502% | 3.311% | 244,355 | 20,000 | 112 s |
| slider_continuous | 768×128 | jittered | 0.200% | 0.753% | 0.611% | 1.129% | 237,955 | 20,000 | 113 s |
| sokoban | 768×512 | lattice | 0.183% | 0.396% | 0.311% | 2.093% | 254,595 | 20,000 | 118 s |
| three_way_switch | 768×640 | lattice | 0.083% | 0.263% | 0.137% | 2.394% | 237,443 | 20,000 | 95 s |
| todo | 640×768 | lattice | 0.095% | 0.937% | 0.479% | 2.839% | 251,523 | 20,000 | 120 s |
| toggle | 768×384 | lattice | 0.028% | 0.231% | 0.124% | 1.504% | 236,675 | 20,000 | 86 s |
| tracks | 768×640 | wide | 0.014% | 0.337% | 0.208% | 1.057% | 478,787 | 40,000 | 433 s |

Three trainings appear in the table. *lattice*: the original recipe, sampling pixel centres of the 4× raster.
*wide*: the same recipe at hidden 192, 5 layers, 40,000 steps (clock, marble, rail, rush_hour, scheduler, tracks;
about 7 GPU minutes each). *jittered*: pixels sampled at a uniformly random sub-pixel offset, with the Fourier
features capped at 6 octaves (a 2-px top wavelength on the 4× raster), 20,000 steps, hidden 128.

## The lattice lesson

A shader trained on pixel centres can be exact at those centres and wrong between them. counter read 0.035% at the
training raster and 1.67% at 0.9× of it: the top Fourier octaves had learned a pattern that happens to agree with the
oracle on the lattice and streaks off it. Every interface with thin strokes or digits did the same (hanoi 3.97%,
reorder 5.04%, slider 3.83%). Sub-pixel jitter during training removes the loophole: the network has to be right
everywhere, which costs it a little exactness on the lattice (0.01–0.05 points) and buys back the off-lattice error
where it was bad. Where the lattice shader was already clean off the lattice (bell, catapult, digit, toggle, …) the
jittered one is marginally worse, so the shipping set keeps whichever of the two scores lower off the lattice, per
interface. The wide shaders were not retrained with jitter yet; clock, scheduler and tracks keep their wide lattice
models because the narrow jittered ones lose to them even off the lattice.

The third column is the *grid* kind (`--kind grid`): two learned feature grids (16 and 64 rows × aspect, 8 channels,
bilinear) in place of most of the Fourier features, feeding a 64-wide, 2-layer MLP. It exists for speed, not
exactness: the MLP shader is latency-bound on the GPU (a chain of dependent texel fetches per pixel, about 30 ms a
frame regardless of resolution on an RTX A4500), and the grid cuts that chain to two layers. It loses on error
everywhere (median 0.58% off the lattice against 0.34%), so it ships only if the idle-GPU timing says the MLP cannot
hold a frame rate at full screen; the 26 are in `models/shaders-grid/` for that test.

| spec | lattice-trained MLP, off-lattice | jittered MLP, off-lattice | grid, off-lattice | ships |
|---|---|---|---|---|
| bell | 0.079% | 0.087% | 0.195% | lattice |
| catapult | 0.016% | 0.021% | 0.062% | lattice |
| clock | 0.157% | 0.371% | 0.459% | wide |
| counter | 1.668% | 0.350% | 0.906% | jittered |
| digit | 0.037% | 0.051% | 0.196% | wide |
| dominoes | 1.258% | 0.195% | 0.231% | jittered |
| fifteen | 0.591% | 0.613% | 0.979% | lattice |
| hanoi | 3.969% | 0.411% | 0.682% | jittered |
| insert_remove | 0.281% | 0.295% | 0.535% | lattice |
| kanban | 0.459% | 0.477% | 0.584% | lattice |
| linked_sliders | 0.546% | 0.522% | 0.942% | jittered |
| marble | 0.089% | 0.101% | 0.345% | wide |
| negotiate | 0.442% | 0.491% | 0.737% | lattice |
| rail | 1.018% | 0.172% | 0.389% | jittered |
| relay | 0.114% | 0.114% | 0.158% | lattice |
| reorder | 5.038% | 1.544% | 1.716% | jittered |
| rush_hour | 0.332% | 0.469% | 1.123% | wide |
| scheduler | 0.215% | 0.330% | 0.951% | wide |
| seesaw | 0.134% | 0.047% | 0.080% | jittered |
| slider | 3.828% | 0.502% | 1.130% | jittered |
| slider_continuous | 1.909% | 0.611% | 1.007% | jittered |
| sokoban | 0.311% | 0.324% | 0.578% | lattice |
| three_way_switch | 0.137% | 0.146% | 0.271% | lattice |
| todo | 0.479% | 0.551% | 1.036% | lattice |
| toggle | 0.124% | 0.143% | 0.133% | lattice |
| tracks | 0.208% | 0.711% | 1.066% | wide |
| **median / worst** | 0.322% / 5.038% | 0.340% / 1.544% | 0.581% / 1.716% | 11 lattice, 9 jittered, 6 wide |


The row for the lattice-trained MLP is the shader that shipped before this round (the wide ones where they exist).

The shader is one network evaluated once per pixel, so the lab draws it as a WebGL fragment shader at the display's
resolution (`crates/nerve/web/index.html`, "shader (GPU)"): 573×286 px in 0.1 ms on a laptop GPU, against 30–100 ms
for a 192-px picture network in WebAssembly. GPU output matches the engine's CPU draw to 0.001/255 per channel
(scripts in this session's record; `scripts/wasm_smoke.mjs` to be extended).

Point-sampling the shader at the scene's raster (no supersampling) is not a fair score: thin lines alias, exactly as
they do for any point sampler, and it reads 2–4% where the supersampled picture reads 0.3–0.6%. The engine reports
both (`mae_native`, `mae_native_point`).

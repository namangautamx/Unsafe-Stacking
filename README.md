# SAM2 Warehouse Goods Segmentation (C++)

Live segmentation of goods (cartons, pallets, stacks, wrapped loads) in warehouse video with
**SAM2**, written in C++ on **ONNX Runtime** (TensorRT/CUDA) + **OpenCV**. Each good gets its own
mask, its own bounding box and a stable ID, shown in a live window. Everything is driven by a YAML
config: the source (video folder, single file, RTSP camera, webcam), the model, thresholds, display
and outputs.

```
source (folder / file / RTSP / webcam)
   │  grabber thread (live: newest frame only, files: paced at native FPS)
   ▼
SAM2 encoder (TensorRT FP16) ──► image embeddings (stay on the GPU)
   │
   ▼
SAM2 decoder (CUDA, batched)  ◄── prompt grid (points or box prompts) per region,
   │                               skipping exclude zones and already-explained pixels
   ▼
filters: SAM2 IoU, stability, area, shape (rectangularity, aspect), exclude zones, NMS
   ▼
YOLO detector (Eqiom model: Person / Forklift / Truck / car / Van) ──► masks that ARE a
   │      detected person or vehicle are dropped; goods on the forks stay
   ▼
CLIP check per mask: "a cardboard box" … vs "a forklift", "a truck", "a person" …
   │      → only goods survive (labels are plain text in the config)
   ▼
stack splitting: big goods get a zoomed SAM2 pass inside their own mask; if it finds
   │      individual boxes, the stack is replaced by them
   │      parts (labels / tape inside one good) → dropped
   ▼
IoU tracker (stable IDs) ──► live window + optional annotated video + JSONL detections
```

## Setup

Requirements: Linux, NVIDIA GPU + CUDA 13 toolkit, `g++` (C++17), `cmake`, OpenCV dev (with
GStreamer/FFmpeg), `libyaml-cpp-dev`, `curl`, `unzip`. TensorRT 10 (`libnvinfer10`) is needed only
for `provider: tensorrt`.

```bash
scripts/setup_deps.sh            # ONNX Runtime + cuDNN + SAM2 tiny models, then builds
scripts/setup_deps.sh tiny small # also fetch other SAM2 sizes: tiny | small | base_plus | large
```

On this machine everything is already in place (`third_party/`, `models/`, `build/`). To rebuild:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Run

```bash
./build/sam2_goods                              # uses config/config.yaml
./build/sam2_goods config/demo_one_mp4.yaml     # sample camera with exclude zones set up
./build/sam2_goods config/rtsp_camera.yaml      # IP camera (edit source.url first)
./build/sam2_goods config/offline_folder.yaml   # no window: annotate every frame to output/
```

The first run with `provider: tensorrt` builds a TensorRT engine (about 1 minute). It is cached
in `trt_cache/`, so later starts take a few seconds.

### Window keys

| key | action | key | action |
|---|---|---|---|
| `q` / `Esc` | quit | `space` / `p` | pause |
| `m` | masks on/off | `b` | boxes on/off |
| `c` | contours on/off | `l` | ID labels on/off |
| `g` | show the prompt grid | `h` | HUD on/off |
| `r` | regions / exclude zones on/off | `n` | next video in the folder |
| `o` | detector boxes on/off | | |
| `s` | save a snapshot (PNG) to `output/snapshots` | | |

## Config files

| file | purpose |
|---|---|
| `config/config.yaml` | main config: video folder from `source.path`, live window, full frame |
| `config/demo_one_mp4.yaml` | example of exclude zones (made for the earlier cold-store camera) |
| `config/rtsp_camera.yaml` | RTSP camera, live window |
| `config/offline_folder.yaml` | sync mode, no window, saves annotated video + detections |

Relative paths are resolved against the config file's directory. Main sections:

* **`source`**: `type: video_folder | video_file | rtsp | webcam`
  * folder/file: `path`, `extensions`, `recursive`, `loop`, `realtime` (play at native FPS), `start_frame`
  * rtsp: `url`, `rtsp_backend: gstreamer | ffmpeg`, `rtsp_transport: tcp | udp`,
    `rtsp_codec: auto | h264 | h265`, `rtsp_latency_ms`, `gstreamer_pipeline` (custom),
    `reconnect_delay_sec`, `max_reconnect_attempts` (0 = forever). A lost camera reconnects on its own.
  * `resize_width`: downscale frames before processing
* **`model`**: `encoder` / `decoder` ONNX paths, then `encoder_provider` and `decoder_provider`
  (`tensorrt | cuda | cpu`, `device_id`, `fp16`, `trt_cache_dir`, ...)
* **`segmentation`**: prompting, quality thresholds, size and shape of a good, stack splitting,
  `regions` (zoomed crops) and `exclude_zones` (see below)
* **`tracker`**: IoU matching, `max_missed`, `smoothing`
* **`processing.mode`**:
  * `async`: the window plays live and SAM2 always works on the newest frame. Use this for cameras.
  * `sync`: each frame waits for its own result. Use this for exact offline annotation; `inference_every_n` skips frames.
* **`display`**: window size, fullscreen, which overlays are drawn, mask alpha
* **`output`**: `save_video`, `save_detections` (JSON lines), `snapshot_dir`, `max_frames`

## Tuning for your camera

SAM2 is class-agnostic: it segments *everything* (forklifts, trucks, people, racks, floor).
Two stages turn that into "goods only, one box per good":

* **`classifier` (CLIP).** Each mask is cropped and compared with the text labels in
  `goods_labels` and `other_labels`. The mask is kept when the summed probability of the goods
  labels is at least `min_goods_prob`. If something is still wrongly kept or dropped, add a label
  that describes it (e.g. `- a hand pallet truck`, `- a wooden crate`) and restart. No retraining
  is needed. Set `show_scores: true` to see each good's probability on screen.
* **`detector` (your Eqiom YOLO, `models/detectors/eqiom_ptfc_v1.onnx`).** It detects people and
  vehicles at about 7 ms per frame. A SAM2 mask is dropped as "that object" when it lies mostly inside
  the object's box (`segmentation.exclude_overlap`) and fills at least `object_min_fill` of it.
  Smaller masks inside the box, like cartons on the forks, are kept. The detected objects are drawn
  in red; press `o` to hide them. The model's own "Goods" class never fires (max score 0.000 over
  75 test frames), so the goods themselves come from SAM2 + CLIP. If the model is retrained with
  working goods labels, its goods boxes could be used as SAM2 box prompts.
* **`segmentation.refine`.** Up to `max_parents` large goods get a second SAM2 pass, zoomed into
  their own mask, that looks for individual boxes. If children covering at least `min_coverage`
  of the stack are found, they replace the stack. Raise `points_per_side` to find smaller boxes
  (slower); lower `min_parent_area_ratio` to also refine smaller stacks.

Further, per camera:

1. **Exclude zones** (most effective). Draw polygons over fixed background: walls, doors, racks,
   docks, machines. No prompts are placed there, so it also runs faster, and masks lying mostly inside
   a zone are dropped. Use normalized `[x, y]` points (0–1) so the zones survive a resolution change.
2. **Size**. `min_area_ratio` / `max_area_ratio` are fractions of the frame. Lower `max_area_ratio`
   drops floors and walls but also whole stacks and pallets.
3. **Shape**. `min_rectangularity` (mask area / min-area rectangle) keeps box-like objects.
4. **Small or far goods**. Raise `points_per_side`, or add `regions` around storage areas. Each region
   is cropped and encoded at full model resolution (one extra encoder pass, about 20 ms each).
5. **Dense stacks of identical cartons**. A single point on a carton in a tight stack makes SAM2
   return the **whole stack** with high confidence. The single-carton mask scores very low, which
   is a known SAM ambiguity. Stacks are then reported as one good. To try splitting them, set
   `prompt_type: both` with `box_sizes` set to the size of one carton in pixels, and turn on
   `suppress_containers`. Results depend a lot on image quality, so check them on your footage.
6. Set `log_level: debug` to see, for each inference, how many masks each filter rejected
   (`iou`, `stability`, `small`, `large`, `excluded`, `nms`, `shape`, `container`, `part`).

`build/sam2_point_probe` shows SAM2's three masks for one prompt, which helps pick thresholds:

```bash
./build/sam2_point_probe config/config.yaml frame.png X Y [crop_x crop_y crop_w crop_h] out.png
PROBE_MODE=box:80:70 ./build/sam2_point_probe ...   # box prompt of 80x70 px around (X, Y)
```

## Outputs

* annotated video (`output.save_video`): what the window shows
* detections (`output.save_detections`), one JSON object per inference:

```json
{"frame_id":12,"source":"one.mp4","source_frame":61,"count":8,"inference_ms":557.1,
 "goods":[{"id":3,"box":[277,314,357,525],"score":0.90,"stability":0.97,"area":149105,
           "rectangularity":0.81,"rotated":[cx,cy,w,h,angle],"missed":0}]}
```

## Performance (RTX 5050 Laptop, 1910×1080, SAM2-tiny)

| stage | time |
|---|---|
| encoder, TensorRT FP16 | ~20 ms (57 ms on the CUDA provider) |
| decoder, CUDA, batch 64 | ~1.9 ms per prompt |
| one full-frame round (576-point grid, ~280–360 prompts decoded) | 0.55–0.9 s |
| with CLIP + stack refinement (busy warehouse view) | ~1.4–1.6 s |

In `async` mode the window stays at the source frame rate (25 fps) and the masks refresh about
1.3–1.8 times per second. The decoder dominates the cost. To make it faster:

* lower `points_per_side`
* add exclude zones (fewer prompts)
* keep `skip_covered_points: true`
* use `multimask: best`

The SAM2 `small`, `base_plus` and `large` encoders are more accurate but slower. The decoder costs
the same for every size.

## Use case: Unsafe Stacking (`logic/`)

The use-case logic of *PRD UC Unsafe Stacking: Max Height & Off-Centre Placement* is in
[`logic/`](logic/README.md). It is plain C++17 with no dependencies and is unit-tested against
every acceptance criterion (`./build/logic/unsafe_stacking_tests`). The app feeds it each
inference: goods boxes are the units, and the detector's people and vehicles are occluders.
It then:
* prints and draws the stacks, layers and `d/W` of the top two layers,
* shows the per-ROI phase (acquiring / idle / settling) in the HUD,
* writes incidents to `unsafe_stacking.incidents_path` as JSON lines, and shows a red banner
  for each one.

Configure it in the `unsafe_stacking:` section: rulebook parameters, calibration, ROI polygons,
and reference dimensions.

## Project layout

```
CMakeLists.txt
config/            YAML configs
include/, src/     config, frame_source (folder/file/RTSP/webcam + grabber thread),
                   sam2_model (ORT encoder/decoder), clip_classifier (C++ CLIP tokenizer + zero-shot),
                   goods_segmenter (prompting, filters, stack refinement),
                   tracker, visualizer, pipeline (async/sync loops, window, outputs), main
tools/             sam2_point_probe (debug tool)
logic/             Unsafe Stacking use-case logic (pure C++17) + unit tests
scripts/           setup_deps.sh
models/            SAM2 ONNX (samexporter format: *.encoder.onnx / *.decoder.onnx), clip/ (CLIP ViT-B/16)
third_party/       ONNX Runtime 1.30 GPU (CUDA 13), cuDNN 9
```

Notes on this SAM2 ONNX export:

* The decoder only broadcasts `has_mask_input` / `mask_input` correctly with batch dimension 1.
  Batched prompts are fed that way.
* The decoder cannot run on TensorRT through ONNX Runtime, because its batch dimension gets frozen.
  It runs on CUDA instead.
# Unsafe-Stacking

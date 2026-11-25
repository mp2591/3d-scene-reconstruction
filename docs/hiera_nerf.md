# Hiera-T + HieraNeRF

A generalizable NeRF with a Hiera-T encoder, a bridge that keeps one embedding
per source photo, and a decoder that attends across photos and along rays.
Code: [`src/scene_recon/models/hiera/`](../src/scene_recon/models/hiera/).
Config: [`configs/models/hiera_nerf.yaml`](../configs/models/hiera_nerf.yaml).

```bash
scene-recon check-model --config configs/hiera_nerf_smoke_test.yaml   # random weights, seconds
scene-recon train --config configs/hiera_nerf_smoke_test.yaml         # CPU, synthetic spheres
scene-recon train --config configs/hiera_nerf_srn_cars.yaml           # needs the weights below
```

## Pretrained weights

The encoder loads the official Hiera-T weights pretrained as a masked
autoencoder on ImageNet-1k (`mae_in1k`; 60% of the 32x32-pixel mask units of
each photo hidden). Download them once and save the state dict:

```python
from pathlib import Path

import torch

Path("weights").mkdir(exist_ok=True)
model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k")
torch.save(model.state_dict(), "weights/hiera_tiny_224_mae_in1k.pth")
```

(`pip install hiera-transformer` provides the same models without Torch Hub.)
The official checkpoint file itself, with its weights under `"model_state"`,
loads too, so it can also be downloaded directly (661 MB) and saved under
the same name:

```bash
mkdir -p weights
curl --fail -L -o weights/download.part https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth \
  && mv weights/download.part weights/hiera_tiny_224_mae_in1k.pth
```

(`--fail` and the temporary name keep an error page from replacing a good file.)

The `state_dict()` of the `timm` Hiera-T loads as well: it uses the same
parameter names. Entries the encoder does not have (stage 4, the
classification head, the MAE decoder) are ignored; a missing or wrongly
shaped entry stops the run with an explanation.

Use a complete file, as above. The first ten blocks of Hiera-S have the same
shapes as Hiera-T's, so the encoder tells the two apart by block 10, the
first block of Hiera-T's fourth stage: its attention must have the fourth
stage's width (Hiera-S continues its third stage there). A file with block 10
but no such evidence is refused. A file cut down to blocks 0 to 9 holds no
evidence either way and is loaded as it is, so do not cut files down: that
would also let a larger Hiera through.

`configs/models/hiera_nerf.yaml` and the suite template's `hiera_mlp` run
load them from `weights/hiera_tiny_224_mae_in1k.pth`, relative to the
directory the command is run from (normally the repository root; `weights/`
is ignored by git). Change `model.encoder.weights` to load them from
elsewhere. If the file is missing, a run stops before training anything, so
a model can't silently start from random weights. For the same reason the
setting is required: an encoder section without `weights` is an error.
`weights: null` starts from random weights on purpose; it is meant for tests,
as in `configs/hiera_nerf_smoke_test.yaml`.

## Encoder: Hiera-T (`hiera`)

| Step | What happens |
|---|---|
| Normalise | ImageNet mean and standard deviation, as the weights expect. |
| Tokens | A 7x7 convolution with stride 4: one 96-channel token per 4x4 pixels, plus the learnt position embedding (resized from the 56x56 grid of pretraining to the image's grid). |
| Stage 1 (1 block) | Attention within mask units: 8x8 tokens = 32x32 pixels. 96 channels, 1/4 resolution. |
| Stage 2 (2 blocks) | Queries max-pooled 2x2; attention within mask units. 192 channels, 1/8. |
| Stage 3 (7 blocks) | Pooled again. The first block, which pools, still attends within mask units (as in the official code, the kind of attention changes one block after the stage); the other six attend globally. 384 channels, 1/16. |
| Stage 4 | Not built: its output is not used. |
| Fuse (adapter) | Stage 2 and 3 outputs resized bilinearly to the 1/4 grid, stacked with stage 1 (672 channels), 1x1 convolution to `num_features`. |

The network is a re-implementation with the module names of the official
code. It follows the official code's token order ("unrolling"), which makes
pooling and mask-unit attention cheap reshapes; `tests/test_hiera_nerf.py`
checks that this order pools correctly and forms 32x32-pixel mask units.

**Image sides must be multiples of 32** (whole mask units). ShapeNet-SRN's
128x128 gives a 32x32 feature map with 4x4 mask units.

## Bridge

For each 3D point and each source photo:

- the point is projected into the photo, and the **feature** (encoder map)
  and **colour** (the photo itself) are read there bilinearly. Hiera's
  feature grid sits 1.5 pixels up and to the left of an even tiling of the
  photo (its first layer is a 7x7 convolution with stride 4 and padding 3),
  which the feature lookup corrects for (`feature_offset`; see
  [writing_a_model.md](writing_a_model.md#where-an-encoders-feature-grid-sits));
- the **direction** from the photo's camera to the point is taken in the
  shared world frame, with its difference to and cosine with the new viewing
  direction;
- the point's **depth** in the photo: its distance from the camera, scaled
  so that the scene bounds near and far (distances along the ray) are 0 and
  1, and whether the photo **sees** the point at all (in front of the camera
  and inside the image).

These are combined by a small MLP into one embedding per photo. The point
itself gets an embedding of its world position (sines and cosines), scaled
by the scene bounds: positions are divided by half the depth range,
(far - near) / 2, which is 0.5 for ShapeNet-SRN. The list of photo
embeddings goes to the decoder unaveraged.

## Decoder (`hiera_ray_attention`)

| Step | Tokens | Result |
|---|---|---|
| 1. Point <-> photos | Self-attention over [point embedding, one embedding per photo], `view_depth` blocks | The point token: the density representation |
| 2. Ray step | `HieraRayTransformer` over the S samples of each ray | Linear layer, softplus: density |
| 3. Colour query -> photos | Query: density representation + new viewing direction. Keys: photo features + directions. Values: photo colours. `color_depth` blocks | Colour representation |
| 4. Ray step | A second `HieraRayTransformer` | Linear layer, sigmoid: RGB |

The Hiera-style ray step: attention within groups of `ray_window_size`
consecutive samples; neighbouring pairs merged into one token; attention
across the half-length ray; each merged token copied back to its two samples
and fused with their local tokens. The position along the ray is added with
sines and cosines before each ray step. It is measured as the signed
distance along the ray from the point nearest the world origin, scaled like
positions, because `query_field` receives points but not ray origins.

Photos that do not see a point are left out of the attention in steps 1 and
3; when no photo sees it, all are used, so the result is always defined.

## Decisions made in the implementation

These points were open in the description and are easy to change:

- **One network for both rendering passes**: `level` is ignored. pixelNeRF
  uses two; the pipeline allows either.
- **Merging along the ray** is a learnt linear map of each pair of samples
  (as in Swin's patch merging), not Hiera's max-pooling.
- **The colour query** uses the output of the density ray step (step 2), not
  the raw output of step 1.
- **The keys of step 3** include the photo's normalised depth along with its
  directions.

## Training

The pipeline already provides what the description asks of rendering and
training: alpha compositing along the ray (`rendering/volume.py`), the
squared-error loss, and end-to-end training. Parameter groups: `encoder` (the
pretrained backbone), `adapter`, `bridge` and `decoder` (from scratch);
`configs/hiera_nerf_srn_cars.yaml` gives the encoder a learning rate of 1e-5
with 2,000 warmup steps and everything else the base rate.

Parts of the description that are pipeline settings or not yet in the pipeline:

- **8 to 12 source photos per target view**: set
  `training.num_source_views: [8, 9, 10, 11, 12]` and `training.views_per_scene`
  to at least 13 (ShapeNet-SRN training scenes have 50 views). The proposal's
  comparison trains with 1 to 3 source views and, for now, evaluates with 1
  and 2 (`evaluation.source_view_sets` in `configs/base.yaml`; a 3-view
  protocol is still to be chosen). Memory in the decoder grows with the
  number of photos, so lower `training.rays_per_scene` if needed.
- **Mixed datasets** (indoor video, camera rigs, phone scans, synthetic
  objects): the pipeline has loaders for ShapeNet-SRN and the synthetic spheres
  only. Each new dataset needs a loader in `src/scene_recon/data/`, and
  mixing them needs a combined training split.
- **Per-scene fine-tuning** is not in the pipeline. Rendering new scenes with
  no extra training is what `scene-recon evaluate` does.

Evaluation renders full images in chunks of `renderer.ray_chunk_size` rays
(2048 in `configs/base.yaml`, the same for every run of a suite); lower it
for all runs if evaluation runs out of GPU memory.

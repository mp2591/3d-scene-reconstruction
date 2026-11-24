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
import torch

model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k")
torch.save(model.state_dict(), "weights/hiera_tiny_224_mae_in1k.pth")
```

(`pip install hiera-transformer` provides the same models without Torch Hub.)
The official checkpoint file itself, with its weights under `"model_state"`,
loads too, and so does the `state_dict()` of the `timm` Hiera-T, which uses
the same parameter names. Entries the encoder does not have (stage 4, the
classification head, the MAE decoder) are ignored; a missing or wrongly
shaped entry stops the run with an explanation.

`configs/models/hiera_nerf.yaml` and the suite template's `hiera_mlp` run
load them from `weights/hiera_tiny_224_mae_in1k.pth`, relative to the
directory the command is run from (normally the repository root; `weights/`
is ignored by git). Change `model.encoder.weights` to load them from
elsewhere. If the file is missing, a run stops before training anything, so
a model can't silently start from random weights. `weights: null` starts
from random weights; it is meant for tests, as in
`configs/hiera_nerf_smoke_test.yaml`.

## Encoder: Hiera-T (`hiera`)

| Step | What happens |
|---|---|
| Normalise | ImageNet mean and standard deviation, as the weights expect. |
| Tokens | A 7x7 convolution with stride 4: one 96-channel token per 4x4 pixels, plus the learnt position embedding (resized from the 56x56 grid of pretraining to the image's grid). |
| Stage 1 (1 block) | Attention within mask units: 8x8 tokens = 32x32 pixels. 96 channels, 1/4 resolution. |
| Stage 2 (2 blocks) | Queries max-pooled 2x2; attention within mask units. 192 channels, 1/8. |
| Stage 3 (7 blocks) | Pooled again; global attention. 384 channels, 1/16. |
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
  and **colour** (the photo itself) are read there bilinearly;
- the **direction** from the photo's camera to the point is taken in the
  shared world frame, with its difference to and cosine with the new viewing
  direction;
- the point's **depth** in the photo, normalised by the scene bounds
  (near, far), and whether the photo **sees** the point at all (in front of
  the camera and inside the image).

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
  comparison trains and evaluates with 1 to 3 views; memory in the decoder grows
  with the number of photos, so lower `training.rays_per_scene` if needed.
- **Mixed datasets** (indoor video, camera rigs, phone scans, synthetic
  objects): the pipeline has loaders for ShapeNet-SRN and the synthetic spheres
  only. Each new dataset needs a loader in `src/scene_recon/data/`, and
  mixing them needs a combined training split.
- **Per-scene fine-tuning** is not in the pipeline. Rendering new scenes with
  no extra training is what `scene-recon evaluate` does.

Evaluation renders full images in chunks of `renderer.ray_chunk_size` rays;
lower it if evaluation runs out of GPU memory.

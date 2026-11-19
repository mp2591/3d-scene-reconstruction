# Configuration

An experiment is described by one YAML file. This page explains how config
files are put together. For the meaning of each setting, read
[`configs/base.yaml`](../configs/base.yaml), which lists and comments all of
them.

## Sections

| Section        | Kind      | Describes                                                     |
|----------------|-----------|---------------------------------------------------------------|
| `experiment`   | settings  | Run name, output directory, random seed, device.              |
| `data`         | component | One dataset each for `train`, `val` and `test`.               |
| `model`        | component | The network.                                                  |
| `loss`         | component | The training loss.                                            |
| `renderer`     | settings  | Samples per ray, background colour, chunk size.               |
| `optimization` | settings  | Optimiser, learning rate, warmup, decay, per-group overrides. |
| `training`     | settings  | Steps, batch size, rays, logging and checkpoint intervals.    |
| `validation`   | settings  | The quick check that runs during training.                    |
| `evaluation`   | settings  | The full test protocol.                                       |
| `profiling`    | settings  | How rendering speed is measured.                              |

**Settings** sections have a fixed set of keys, defined as dataclasses in
[`config.py`](../src/scene_recon/config.py). A misspelt or missing key stops
the run with a message naming the key.

**Component** sections select a class by `name`. All other keys are passed to
that class's constructor, so the valid keys are whatever its `__init__`
accepts:

```yaml
model:
  name: tiny_pixel_aligned_nerf   # looked up in the MODELS registry
  encoder: {...}                  # passed to TinyPixelAlignedNeRF.__init__
  decoder: {...}
```

A key named `name` always means "select a component". Do not use it for
anything else.

## Composing configs

`base` names one or more files to load first. Paths are relative to the file
that contains the key, and later files override earlier ones:

```yaml
base:
  - base.yaml                  # pipeline settings
  - data/srn_cars.yaml         # the data section
  - models/my_model.yaml       # the model section

experiment:
  run_name: my_experiment      # overrides the value from base.yaml

training:
  batch_size: 8                # overrides one key; the rest of `training` is kept
```

Sections are merged key by key. There is one exception: when a component
section selects a different `name` than its base, it replaces the base section
entirely, because the arguments of one class mean nothing to another.

## Overriding from the command line

```bash
scene-recon train --config configs/smoke_test.yaml --set training.num_steps=50 model.decoder.hidden_width=32
```

Keys are dotted paths. Values are parsed as YAML, so numbers, booleans, lists
(`evaluation.metrics=[psnr,ssim]`) and `null` work as they do in a file.

## Several runs in one file

A suite file combines a shared config fragment with one fragment per run, and
runs them all; see [suites.md](suites.md). Each fragment is written exactly
like the files described on this page.

## Rules beyond spelling

Every value is checked against its declared type and against value rules,
and all problems are reported at once:

* **Types.** Whole numbers must be whole numbers (`2.9` and `true` are
  rejected; `4.0e+5` is accepted as 400000), true/false settings must be
  `true` or `false`, and lists must contain the right kind of values.
* **Ranges.** Counts and intervals must be meaningful: at least one training
  step, at least one evaluated scene, a positive learning rate, and so on.
  The minimums are listed in `_VALUE_RULES` in
  [`config.py`](../src/scene_recon/config.py).
* **Samples per ray.** At least 2 coarse samples, and at least 3 when the
  fine pass is enabled (fine samples are drawn between coarse samples).
* **View lists.** Source view lists must be non-empty and must not repeat a
  view. Each entry of `evaluation.source_view_sets` must have a different
  number of views: results are labelled and plotted by the number of input
  views, so two sets of the same size could not be told apart.
* **Metrics.** `validation.metrics` and `evaluation.metrics` list metric
  names, or mappings with a `name` and constructor options, such as
  `{name: lpips, network: alex}`. No metric may be listed twice. Whether the
  options fit the metric is checked when it is built, before any training.
* **Per-group overrides.** Each entry of `optimization.group_overrides` may
  set `learning_rate` (a finite positive number) and `warmup_steps` (a whole
  number, zero or more), checked like the settings they override.

## Reproducing a run

Each run saves its fully merged config to `runs/<run_name>/config.yaml`. That
file has no `base` and can be passed to `--config` directly.

## No hidden defaults

The settings dataclasses have no default values. Every setting the pipeline
reads is written in `configs/base.yaml`, so the YAML files are the only place
to look when asking what value a run used. When you add a setting to a
dataclass, add it to `base.yaml` as well; the error message for a missing key
will remind you.

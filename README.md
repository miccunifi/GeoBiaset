# GeoBiaset

This repository contains the code to generate the GeoBiaset image set and to run geolocation experiments on it. The minimal public path is:

1. install the environment;
2. place the released dataset under `GeoBiaset/`;
3. optionally run the generation pipeline;
4. evaluate GeoCLIP or Qwen3 and inspect the metrics.

This public repository is intended to contain code and configuration examples. It does not include the original source image datasets, the FairFace-derived face split, or generated experiment outputs unless those outputs are explicitly released separately.

## 1. Installation

Install `uv` if needed:

```bash
pip install uv
```

Create the environment:

```bash
uv sync
```

If you prefer pip, install the project dependencies from `pyproject.toml` in a Python 3.12 environment.

Generation and Qwen3 evaluation require a CUDA-capable GPU. GeoCLIP can run on CPU, but GPU is recommended. The first run can download model weights from Hugging Face, so make sure the machine has internet access and, when needed, Hugging Face access to the requested checkpoints.

## 2. Set Up

Put the released dataset in this layout:

```text
GeoBiaset/
  images/
  data.csv
```

The released data file `GeoBiaset/data.csv` should contain these columns:

```text
image_path,background_path,lat,lon,scene_type,scene_attribute,inserted_race
```

`image_path` and `background_path` should resolve under `GeoBiaset/`. For example:

```text
images/example.jpg
```

For evaluation, provide a manifest CSV with these columns:

```text
source_path,dataset_path,lat,lon,category,augmentation
```

`dataset_path` should point to the image evaluated by the model. `source_path` should identify the corresponding ground-truth/background image and include the source region when POPI metrics are needed, for example `balanced_dataset/white/images/example_gt.jpg`. Ground-truth rows should use `augmentation=ground truth`; edited rows should use the inserted group label such as `White`, `Black`, `East Asian`, `Indian`, `Latino_Hispanic`, `Middle Eastern`, or `Southeast Asian`.

Use this config for evaluation:

```yaml
source:
  name: GeoBiaset
  root_dir: .

dataset:
  name: GeoBiaset
  root_dir: GeoBiaset
  manifest_csv: GeoBiaset/eval_manifest.csv
```

Save it as `config_geobiaset.yaml`.

For the generation pipeline, the repo expects users to prepare their own input images and face references. During development, face references were selected from the FairFace dataset, following the dataset's intended research use and metadata structure. The original source image pools and derived face reference split are not released. For ethical reasons, privacy reasons, and to avoid enabling discriminatory use, we do not disclose or redistribute the face split or the exact Im2GPS3k/YFCC4k-derived splits.

To run generation, prepare:

- a background image set split into geographical regions;
- a face reference set split across gender, age, and race categories.

The pipeline reads those prepared inputs from:

```text
balanced_dataset/filtered_images.csv
faces/selected_faces.csv
```

It also needs the background images referenced by `balanced_dataset/filtered_images.csv` and the face images referenced by `faces/selected_faces.csv`. Users are responsible for creating these CSVs and ensuring they have the rights and ethical approval to use the images.

## 3. Generation Pipeline

Run the generation pipeline:

```bash
uv run python pipeline.py
```

To choose the metadata output path:

```bash
uv run python pipeline.py --dataset_csv dataset/dataset_metadata.csv
```

The pipeline writes generated samples and masks to:

```text
dataset/images/
dataset/masks/
```

and writes metadata CSV files under `dataset/`. The generated metadata can be used as an evaluation manifest after checking that its image paths resolve correctly from the configured dataset root.

## 4. Run Evaluation On GeoCLIP Or Qwen3

GeoCLIP is the default lightweight experiment:

```bash
uv run python evaluate_models.py --config config_geobiaset.yaml --models geoclip --output-csv GeoBiaset/geoclip_results.csv
```

Qwen3 can also be run from Hugging Face, but it is much heavier:

```bash
uv run python evaluate_models.py --config config_geobiaset.yaml --models qwen3-vl-8b --output-csv GeoBiaset/qwen3_results.csv
```

To add POPI columns and image quality metrics to a result CSV:

```bash
uv run append-evaluation-metrics GeoBiaset/geoclip_results.csv --output GeoBiaset/geoclip_results_with_extra_metrics.csv
```

POPI requires matching ground-truth rows in the evaluation output, because it compares each generated/injected row against the corresponding ground-truth prediction.

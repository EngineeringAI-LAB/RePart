# RePart

[Project Page](https://engineeringai-lab.github.io/RePart/)

## Setup

Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

Provide a mesh directory with UID-named files (for example,
`/path/to/meshes/<uid>.obj`) and a matching PartNet point-sample
directory:

```text
/path/to/partnet_points/
`-- <uid>/
    `-- point_sample/
        |-- label-10000.txt
        |-- ply-10000.ply
        `-- pts-10000.txt
```

The mesh filename stem must match the UID directory. Meshes can be OBJ, PLY,
STL, OFF, GLB, or GLTF. To use precomputed SDF CSV files, pass `--sdf-root`
to the training script; otherwise SDFs are generated from the input meshes.

## Train

```bash
bash scripts/train_category.sh --data-root /path/to/meshes --point-sample-root /path/to/partnet_points --out-dir /path/to/category_run --rollout-episodes-per-epoch 100
bash scripts/train_joint.sh --run-dir /path/to/little_run --run-dir /path/to/container_run --run-dir /path/to/furniture_run --out-dir /path/to/joint_run
```

## Evaluate

```bash
bash scripts/evaluate.sh --checkpoint /path/to/joint_run/sq_partnet_rl_epoch_0100.pt --data-root /path/to/test_meshes --point-sample-root /path/to/partnet_points --out-dir /path/to/eval_run
```

The output directory contains grouped SQs and projected mesh labels.

## Tests

```bash
python -m unittest discover -s tests
```

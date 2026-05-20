# C++ Static Pose Optimizer

This folder contains a C++ port of `static_pose_optimizer_ba.py` and a sample application that runs two examples:

1. `Random test example` using synthetic robot poses and 2D observations.
2. `Dataset example` reading real data from a configurable dataset path.

## Build

```bash
source /home/byd/miniforge3/bin/activate cv48
mkdir -p cpp_static_pose_optimizer/build
cd cpp_static_pose_optimizer/build
cmake ..
make -j4
```

## Run

```bash
./static_pose_optimizer_example --run_dataset --dataset_path=/home/byd/work/socket-pose-estimator/dataset/save_data3/chb_20260511_120244 --max_frames=10
```

The program automatically runs the random test example. The dataset example uses:

- `camPrms/cam_intrisic.xml` for camera matrix and distortion coefficients
- `camPrms/cam_2_gripper.xml` for the hand-eye extrinsic
- `data/*.txt` for 2D feature coordinates
- `data/*.npy` for robot end-effector poses

## Notes

- The `.npy` loader currently supports little-endian `float64` arrays with `fortran_order=False`.
- The optimizer performs a Gauss-Newton / Levenberg-style update on a global object pose plus per-frame SE(3) perturbations.

#!/usr/bin/env python3
"""Convert recorded ARKit trajectory to robot-executable 6D pose sequence.

Usage:
    python convert_trajectory.py recordings/trajectory_xxx.npy --target_hz 10 --remap arkit_to_ros --plot

Output:
    recordings/trajectory_xxx_robot.npy   # {"timestamps": (M,), "poses": (M,6)}
    recordings/trajectory_xxx_robot.csv   # timestamp, x, y, z, rx, ry, rz
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Pose utilities (inlined from UMI pose_util.py / interpolation_util.py)
# ---------------------------------------------------------------------------

def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous matrix to 6D pose [x, y, z, rx, ry, rz] (axis-angle)."""
    pos = mat[:3, 3]
    rotvec = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert 6D pose [x, y, z, rx, ry, rz] to 4x4 homogeneous matrix."""
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat


class PoseInterpolator:
    """Interpolate SE(3) poses: linear for translation, SLERP for rotation."""

    def __init__(self, timestamps: np.ndarray, transforms: np.ndarray):
        """
        Args:
            timestamps: (N,) monotonic timestamps
            transforms: (N, 4, 4) homogeneous transforms
        """
        assert len(timestamps) == len(transforms)
        self.timestamps = timestamps
        self.positions = transforms[:, :3, 3]  # (N, 3)
        self.rotations = Rotation.from_matrix(transforms[:, :3, :3])
        self.slerp = Slerp(timestamps, self.rotations)

    def __call__(self, query_times: np.ndarray) -> np.ndarray:
        """Interpolate at query_times. Returns (M, 4, 4) transforms."""
        # Clamp to valid range
        t_min, t_max = self.timestamps[0], self.timestamps[-1]
        query_times = np.clip(query_times, t_min, t_max)

        # Interpolate position (linear)
        positions = np.column_stack([
            np.interp(query_times, self.timestamps, self.positions[:, i])
            for i in range(3)
        ])

        # Interpolate rotation (SLERP)
        rotations = self.slerp(query_times)

        # Assemble 4x4 transforms
        N = len(query_times)
        transforms = np.zeros((N, 4, 4))
        transforms[:, :3, :3] = rotations.as_matrix()
        transforms[:, :3, 3] = positions
        transforms[:, 3, 3] = 1.0
        return transforms


# ---------------------------------------------------------------------------
# Coordinate remap presets
# ---------------------------------------------------------------------------

# ARKit: X-right, Y-up, Z-toward-viewer (camera looking -Z)
# ROS REP-103: X-forward, Y-left, Z-up

REMAP_PRESETS = {
    "identity": np.eye(3),
    "arkit_to_ros": np.array([
        [0, 0, -1],   # ROS X = ARKit -Z (forward)
        [-1, 0, 0],   # ROS Y = ARKit -X (left)
        [0, 1, 0],    # ROS Z = ARKit  Y (up)
    ], dtype=float),
}


def apply_remap(transforms: np.ndarray, R_remap: np.ndarray) -> np.ndarray:
    """Apply coordinate remap to all transforms.

    Given remap rotation R, each transform T becomes:
        T' = R_hom @ T @ R_hom^{-1}
    This rotates both the position and orientation into the new frame.
    """
    R_hom = np.eye(4)
    R_hom[:3, :3] = R_remap
    R_hom_inv = np.eye(4)
    R_hom_inv[:3, :3] = R_remap.T

    out = np.zeros_like(transforms)
    for i in range(len(transforms)):
        out[i] = R_hom @ transforms[i] @ R_hom_inv
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_trajectory(path: str):
    """Load .npy trajectory file."""
    data = np.load(path, allow_pickle=True).item()
    timestamps = data["timestamps"]
    transforms = data["transforms"]
    print(f"Loaded {len(timestamps)} frames, duration={timestamps[-1] - timestamps[0]:.1f}s")
    return timestamps, transforms


def normalize_to_relative(transforms: np.ndarray) -> np.ndarray:
    """Make first frame identity: T_rel_i = T_0^{-1} @ T_i."""
    T0_inv = np.linalg.inv(transforms[0])
    return np.array([T0_inv @ T for T in transforms])


def resample(timestamps: np.ndarray, transforms: np.ndarray,
             target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample trajectory to target frequency using SLERP interpolation."""
    t_start, t_end = timestamps[0], timestamps[-1]
    duration = t_end - t_start
    n_samples = max(2, int(duration * target_hz) + 1)
    query_times = np.linspace(t_start, t_end, n_samples)

    interp = PoseInterpolator(timestamps, transforms)
    resampled_transforms = interp(query_times)

    actual_hz = (n_samples - 1) / duration if duration > 0 else target_hz
    print(f"Resampled: {len(timestamps)} -> {n_samples} frames ({actual_hz:.1f} Hz)")
    return query_times, resampled_transforms


def transforms_to_poses(transforms: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) transforms to (N, 6) axis-angle poses."""
    return np.array([mat_to_pose(T) for T in transforms])


def save_output(base_path: str, timestamps: np.ndarray, poses: np.ndarray):
    """Save robot pose sequence as .npy and .csv."""
    npy_path = base_path + "_robot.npy"
    np.save(npy_path, {"timestamps": timestamps, "poses": poses})
    print(f"Saved {npy_path}  ({len(poses)} poses)")

    csv_path = base_path + "_robot.csv"
    with open(csv_path, 'w') as f:
        f.write("timestamp,x,y,z,rx,ry,rz\n")
        for i in range(len(poses)):
            vals = [timestamps[i]] + poses[i].tolist()
            f.write(",".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"Saved {csv_path}")


def plot_trajectory(timestamps: np.ndarray, poses: np.ndarray, title: str = ""):
    """3D plot of trajectory with start/end markers."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = poses[:, 0], poses[:, 1], poses[:, 2]

    # Color by time
    colors = np.linspace(0, 1, len(poses))
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', s=2, alpha=0.7)
    ax.plot(xs, ys, zs, 'k-', alpha=0.15, linewidth=0.5)

    # Mark start and end
    ax.scatter(*poses[0, :3], color='green', s=100, marker='o', label='Start')
    ax.scatter(*poses[-1, :3], color='red', s=100, marker='x', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title or "Robot Trajectory (relative)")
    ax.legend()

    # Equal aspect ratio
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2.0
    mid = np.array([(xs.max() + xs.min()) / 2,
                     (ys.max() + ys.min()) / 2,
                     (zs.max() + zs.min()) / 2])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.colorbar(scatter, label='Time (normalized)', shrink=0.6)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Convert ARKit trajectory to robot 6D pose sequence")
    parser.add_argument("input", help="Path to recorded .npy trajectory file")
    parser.add_argument("--target_hz", type=float, default=None,
                        help="Resample to target frequency (Hz). Default: keep original.")
    parser.add_argument("--remap", type=str, default="identity",
                        choices=list(REMAP_PRESETS.keys()),
                        help="Coordinate remap preset (default: identity)")
    parser.add_argument("--plot", action="store_true", help="Show 3D trajectory plot")
    args = parser.parse_args()

    # 1. Load
    timestamps, transforms = load_trajectory(args.input)

    # 2. Normalize to relative motion (T_0 = I)
    transforms = normalize_to_relative(transforms)
    print(f"Normalized to relative motion (first frame = identity)")

    # 3. Coordinate remap
    R_remap = REMAP_PRESETS[args.remap]
    if not np.allclose(R_remap, np.eye(3)):
        transforms = apply_remap(transforms, R_remap)
        print(f"Applied coordinate remap: {args.remap}")

    # 4. Resample (optional)
    if args.target_hz is not None:
        timestamps, transforms = resample(timestamps, transforms, args.target_hz)

    # 5. Convert to 6D poses
    poses = transforms_to_poses(transforms)
    print(f"Converted to 6D poses: shape={poses.shape}")

    # 6. Save
    base_path = os.path.splitext(args.input)[0]
    save_output(base_path, timestamps, poses)

    # 7. Plot
    if args.plot:
        plot_trajectory(timestamps, poses,
                        title=f"Trajectory ({args.remap}, {len(poses)} frames)")


if __name__ == '__main__':
    main()

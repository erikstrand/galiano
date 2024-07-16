from pathlib import Path
import click
import numpy as np


npz_dir = Path("data/npz")
npz_small_dir = Path("data/npz_small")


def main():
    base_names = [
        "bc_092b084_3_2_3_xyes_8_utm10_2019",
        "bc_092b084_3_2_4_xyes_8_utm10_2019",
        "bc_092b084_3_4_1_xyes_8_utm10_2019",
        "bc_092b084_3_4_2_xyes_8_utm10_2019",
    ]

    print("Loading points...")
    # ll = lower left, lr = lower right, ul = upper left, ur = upper right
    points_ll = np.load(npz_dir / f"{base_names[0]}_xyz.npz")["points"]
    points_lr = np.load(npz_dir / f"{base_names[1]}_xyz.npz")["points"]
    points_ul = np.load(npz_dir / f"{base_names[2]}_xyz.npz")["points"]
    points_ur = np.load(npz_dir / f"{base_names[3]}_xyz.npz")["points"]
    points = np.concatenate([points_ll, points_lr, points_ul, points_ur])
    n_points = points.shape[0]
    print(f"Loaded {n_points} points.")
    print(points.dtype)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    print(mins)
    print(maxs)
    print("")

    print("Loading classifications...")
    class_ll = np.load(npz_dir / f"{base_names[0]}_class.npz")["classifications"]
    class_lr = np.load(npz_dir / f"{base_names[1]}_class.npz")["classifications"]
    class_ul = np.load(npz_dir / f"{base_names[2]}_class.npz")["classifications"]
    class_ur = np.load(npz_dir / f"{base_names[3]}_class.npz")["classifications"]
    classes = np.concatenate([class_ll, class_lr, class_ul, class_ur])
    print(f"Loaded {classes.shape[0]} classifications.")
    print(classes.dtype)
    print("")

    tree_class = 1
    ground_class = 2
    noise_class = 7
    print(f"Extracting trees and ground...")
    tree_mask = classes == tree_class
    tree_points = points[tree_mask]
    n_tree_points = tree_points.shape[0]
    print(f"{n_tree_points} tree points")
    ground_mask = classes == ground_class
    ground_points = points[ground_mask]
    n_ground_points = ground_points.shape[0]
    print(f"{n_ground_points} ground points")
    print("")

    print("Collecting a random subset of tree points...")
    max_subset_points = 1_000_000
    n_points_subset = min(max_subset_points, n_tree_points)
    tree_indices = np.arange(n_tree_points)
    tree_indices = np.random.choice(tree_indices, size=n_points_subset, replace=False)
    tree_points_subset = tree_points[tree_indices]
    print("")

    print("Collecting a random subset of ground points...")
    n_points_subset = min(max_subset_points, n_ground_points)
    ground_indices = np.arange(n_ground_points)
    ground_indices = np.random.choice(ground_indices, size=n_points_subset, replace=False)
    ground_points_subset = ground_points[ground_indices]
    print("")

    print("Saving to npz_small...")
    np.savez(npz_small_dir / "tree_points.npz", points=tree_points_subset)
    np.savez(npz_small_dir / "ground_points.npz", points=ground_points_subset)


if __name__ == "__main__":
    main()

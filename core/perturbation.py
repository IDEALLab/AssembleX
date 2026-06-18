import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh


def resolve_collision(
    meshes: list[trimesh.Trimesh],
    moving_part: trimesh.Trimesh,
    alpha: float = 1.0,
    beta: float = 0.1,
    lr: float = 0.4,
    epsilon: float = 1e-3,
    max_iters: int = 100,
    overlap_tol: float = 1e-3,
    gradient_tol: float = 5e-3,
    draw: bool = False,
):
    """
    Moves a specified CAD part to minimize collision overlap and translation distance
    using finite-differences gradient descent.

    Args:
        meshes: List of meshes representing the assembly.
        moving_part: The trimesh object to be moved.
        alpha: Weight for the overlap penalty.
        beta: Weight for the distance penalty.
        lr: Learning rate (step size multiplier).
        epsilon: Perturbation amount for finite differences.
        max_iters: Maximum number of descent steps.
        overlap_tol: Tolerance to consider overlap as "zero".
        gradient_tol: Tolerance to consider gradient as "zero".
        draw: Whether to plot the optimization path.
    Returns:
        tuple: (Total translation vector, Modified trimesh object)
    """
    start_time = time.time()

    # 0. Setup and save original position
    move_mesh = (
        moving_part.copy()
    )  # Work with a copy to avoid modifying original until final
    original_com = moving_part.center_mass.copy()
    total_translation = np.zeros(3)
    union_mesh = trimesh.boolean.union(meshes)
    if draw:
        pv_mesh = pv.wrap(union_mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.5, label="Assembly")
        plotter.add_mesh(
            pv.wrap(move_mesh), color="yellow", opacity=0.7, label="Moving Part"
        )
        plotter.show()
    scaling_factor = np.cbrt(
        move_mesh.volume
    )  # Use cube root of volume to get a characteristic length scale

    if draw:
        # Lists to record path and costs
        x_path: list[float] = []
        y_path: list[float] = []
        z_path: list[float] = []
        costs: list[float] = []
        overlaps: list[float] = []
        normed_overlaps: list[float] = []

        def plot_results():
            _fig, axes = plt.subplots(3, 2, figsize=(12, 12))

            axes[0, 0].plot(x_path, marker="o")
            axes[0, 0].set_title("X Axis Translation Path")
            axes[0, 0].set_ylabel("X Translation")

            axes[0, 1].plot(y_path, marker="o")
            axes[0, 1].set_title("Y Axis Translation Path")
            axes[0, 1].set_ylabel("Y Translation")

            axes[1, 0].plot(z_path, marker="o")
            axes[1, 0].set_title("Z Axis Translation Path")
            axes[1, 0].set_ylabel("Z Translation")

            all_paths = x_path + y_path + z_path
            if all_paths:
                y_min, y_max = min(all_paths), max(all_paths)
                if y_min != y_max:
                    axes[0, 0].set_ylim(y_min, y_max)
                    axes[0, 1].set_ylim(y_min, y_max)
                    axes[1, 0].set_ylim(y_min, y_max)

            axes[1, 1].plot(costs, marker="o", color="red")
            axes[1, 1].set_title("Recorded Cost")
            axes[1, 1].set_ylabel("Cost")

            axes[2, 0].plot(overlaps, marker="o", color="orange")
            axes[2, 0].set_title("Recorded Overlap")
            axes[2, 0].set_ylabel("Overlap")
            axes[2, 0].set_xlabel("Iteration")

            axes[2, 1].plot(normed_overlaps, marker="o", color="purple")
            axes[2, 1].set_title("Recorded Normed Overlap")
            axes[2, 1].set_ylabel("Normed overlap")
            axes[2, 1].set_xlabel("Iteration")

            plt.tight_layout()
            plt.show()

    for iteration in range(max_iters):
        # Get distance and overlap at current position
        current_overlap = get_overlap(
            meshes, move_mesh, union_mesh=union_mesh, mode="boolean"
        )
        current_com = move_mesh.center_mass
        distance_vec = current_com - original_com
        distance = np.linalg.norm(distance_vec)

        # cost function
        current_cost = (alpha * (current_overlap / scaling_factor**3) ** 2) + (
            beta * (distance / scaling_factor) ** 2
        )

        if draw:
            # Record current state
            x_path.append(total_translation[0])
            y_path.append(total_translation[1])
            z_path.append(total_translation[2])
            costs.append(current_cost)
            overlaps.append(current_overlap)
            normed_overlaps.append(current_overlap / scaling_factor**3)

        """
        # 3.1 Exit condition: if overlap is essentially 0
        if current_overlap / scaling_factor**3 <= overlap_tol:
            total_time = time.time() - start_time
            print(f"Converged in {iteration} iterations. Overlap resolved.")
            print(f"resolve_collision total time: {total_time:.6f} seconds")
            if draw:
                plot_results()
            return total_translation, move_mesh
        """

        # Find gradient via finite differences
        gradient = np.zeros(3)

        for i in range(3):  # For X, Y, Z axes
            perturb_vec = np.zeros(3)
            perturb_vec[i] = epsilon

            # Slightly perturb part
            move_mesh.apply_translation(perturb_vec)

            overlap_pert = get_overlap(meshes, move_mesh, union_mesh=union_mesh)
            dist_pert = np.linalg.norm(move_mesh.center_mass - original_com)
            cost_pert = (alpha * (overlap_pert / scaling_factor**3) ** 2) + (
                beta * (dist_pert / scaling_factor) ** 2
            )

            # Revert the perturbation
            move_mesh.apply_translation(-perturb_vec)

            # Calculate partial derivative
            gradient[i] = (cost_pert - current_cost) / epsilon

        grad_norm = np.linalg.norm(gradient)
        # Termination if gradient is very small (local minimum)
        if grad_norm < gradient_tol:
            total_time = time.time() - start_time
            print(
                f"Converged in {iteration} iterations. Local minimum found, overlap: {current_overlap:.6f}."
            )
            print(f"resolve_collision total time: {total_time:.6f} seconds")
            if draw:
                plot_results()
            return True, total_translation, move_mesh

        # Basic gradient descent step with learning rate and scaling
        # step = -lr * (gradient / grad_norm)*scaling_factor
        step = -lr * gradient * scaling_factor

        # Apply step
        move_mesh.apply_translation(step)
        total_translation += step

        print(
            f"Iteration {iteration + 1}: Overlap={current_overlap:.6f}, Distance={distance:.6f}, Cost={current_cost:.6f}, Step={step}, Gradient Norm={grad_norm:.6f}"
        )

    total_time = time.time() - start_time
    print(
        f"Maximum iterations ({max_iters}) reached. Remaining overlap: {current_overlap:.4f}"
    )
    print(f"resolve_collision total time: {total_time:.2f} seconds")
    if draw:
        plot_results()
    return False, total_translation, move_mesh


def get_overlap(meshes, move_mesh, mode="boolean", union_mesh=None):
    time.time()

    if not union_mesh:
        union_mesh = trimesh.boolean.union(meshes)

    manager = trimesh.collision.CollisionManager()
    overlap = 0

    for i, mesh in enumerate(meshes):
        manager.add_object(name=f"mesh_{i}", mesh=mesh)

    is_overlapping, contact_data = manager.in_collision_single(
        move_mesh, return_data=True
    )
    overlap = 0

    if is_overlapping:
        if mode == "depth":
            overlap = max(contact_data["depth"])

        elif mode == "boolean":
            overlap_mesh = move_mesh.intersection(union_mesh)
            overlap = overlap_mesh.volume

    # print(f"get_overlap execution time: {time.time() - start_time:.6f} seconds, mode: {mode}, overlap: {overlap:.6f}")
    return overlap

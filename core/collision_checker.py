import matplotlib.pyplot as plt
import trimesh


class CollisionChecker:
    def __init__(self, assembly):
        self.assembly = assembly

    def get_collisions(self, draw=False, mode="boolean", show=False):
        if mode not in ("depth", "boolean"):
            raise ValueError(
                "Invalid mode. Mode should be either 'depth' for depth-based collision checking or 'boolean' for boolean intersection-based collision checking."
            )

        overlap_threshold = 0.05
        depth_threshold = 0.1
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print("\nChecking for static collisions...")

        manager = trimesh.collision.CollisionManager()

        for obj in self.assembly.objects.values():
            if not obj.tri_mesh.is_watertight:
                print(
                    f"Warning: Part {obj.id} is not watertight. Ending collision check."
                )
                return set(), False
            manager.add_object(name=obj.id, mesh=obj.tri_mesh)

        is_overlapping, names_overlapping, collision_data = (
            manager.in_collision_internal(return_names=True, return_data=True)
        )

        all_collisions = []
        major_collisions = set()
        max_depth = {}

        if mode == "boolean":
            for obj1, obj2 in names_overlapping:
                overlap_mesh = self.assembly.objects[obj1].tri_mesh.intersection(
                    self.assembly.objects[obj2].tri_mesh
                )
                v1 = (
                    self.assembly.objects[obj1].tri_mesh.volume
                    if self.assembly.objects[obj1].tri_mesh.is_volume
                    else 0
                )
                v2 = (
                    self.assembly.objects[obj2].tri_mesh.volume
                    if self.assembly.objects[obj2].tri_mesh.is_volume
                    else 0
                )
                # v_tot = v1 + v2
                v_min = min(v1, v2)
                # overlap_ratio = overlap_mesh.volume / v_tot if v_tot > 0 else 0
                overlap_ratio = overlap_mesh.volume / v_min if v_min > 0 else 0
                if overlap_ratio >= overlap_threshold:
                    major_collisions.add((obj1, obj2))
                all_collisions.append(overlap_ratio)

        elif mode == "depth":
            for contact_obj in collision_data:
                obj1, obj2 = contact_obj.names
                depth = contact_obj.depth
                if (obj1, obj2) not in max_depth or depth > max_depth[(obj1, obj2)]:
                    max_depth[(obj1, obj2)] = depth
                all_collisions.append(depth)

            for (obj1, obj2), depth in max_depth.items():
                if depth >= depth_threshold:
                    major_collisions.add((obj1, obj2))

        if major_collisions:
            print(
                f"Major collisions detected between the following parts: {major_collisions}"
            )
            if (
                self.assembly.evaluation
                and self.assembly.evaluation.verbose
                and mode == "depth"
            ):
                print("Detailed maximum penetration depths:")
                for (o1, o2), d in max_depth.items():
                    if d >= depth_threshold:
                        print(f"  {o1} - {o2}: {d:.6f}")
        elif is_overlapping:
            print("Overlap detected but no major collisions found.")
        elif self.assembly.evaluation and self.assembly.evaluation.verbose:
            print("No overlaps detected.")

        if draw:
            plt.figure(figsize=(10, 6))
            if mode == "boolean":
                plt.hist(
                    [100 * x for x in all_collisions],
                    bins=200,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
                plt.axvline(
                    x=100 * overlap_threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Major Overlap Threshold",
                )
                plt.xlabel("% of sum of two part volumes that is overlapping")
                plt.ylabel("Frequency")
                plt.title("Distribution of Part Overlaps")
                plt.legend()
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.xlim(left=0)
                plt.tight_layout()
                plt.savefig(self.assembly.output_dir / "overlap_distribution.png")
                plt.close()
            elif mode == "depth":
                plt.hist(
                    all_collisions, bins=200, alpha=0.7, color="blue", edgecolor="black"
                )
                plt.axvline(
                    x=depth_threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Major Collision Threshold",
                )
                plt.xlabel("Penetration Depth (units of the mesh)")
                plt.ylabel("Frequency")
                plt.title("Distribution of Penetration Depths")
                plt.legend()
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.xlim(left=0)

                if max_depth:
                    plt.hist(
                        list(max_depth.values()),
                        bins=200,
                        alpha=0.5,
                        color="orange",
                        edgecolor="black",
                        label="Max Penetration Depths",
                    )
                    plt.legend()
                plt.yscale("log")
                plt.tight_layout()
                plt.savefig(
                    self.assembly.output_dir / "penetration_depth_distribution.png"
                )
                plt.close()

        self.assembly.renderer.view_collisions(major_collisions, show=show, save=True)

        return major_collisions, all_collisions, True

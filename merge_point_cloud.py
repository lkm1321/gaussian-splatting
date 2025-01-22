import open3d as o3d
import fire
import os

def main(
    path,
    num_files,
    downsample_ratio=0.05,
    output_file="points3d.ply"
):
    merged_pc = sum(
        (o3d.io.read_point_cloud(
            os.path.join(path, f'{idx:06}.ply')
        ).random_down_sample(downsample_ratio) for idx in range(num_files)), 
        o3d.geometry.PointCloud()
    )
    merged_ply = o3d.geometry.TriangleMesh(vertices=merged_pc.points, triangles=o3d.utility.Vector3iVector([]))
    merged_ply.compute_vertex_normals()
    if merged_pc.has_colors():
        merged_ply.vertex_colors = merged_pc.colors
    else:
        merged_ply.paint_uniform_colors(np.array([1., 1., 1.]))
    o3d.io.write_triangle_mesh(output_file, merged_ply)

if __name__ == "__main__":
    fire.Fire(main)

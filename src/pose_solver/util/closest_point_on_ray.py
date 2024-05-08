from typing import List


# Find the closest point on a ray in 3D
def closest_point_on_ray(
    ray_source: List[float],
    ray_direction: List[float],
    query_point: List[float],
    forward_only: bool
):
    # Let ray_point be the closest point between query_point and the ray.
    # (ray_point - query_point) will be perpendicular to ray_direction.
    # Let ray_distance be the distance along the ray where the closest point is.
    # So we have two equations:
    #     (1)    (ray_point - query_point) * ray_direction = 0
    #     (2)    ray_point = ray_source + ray_distance * ray_direction
    # If we substitute eq (2) into (1) and solve for ray_distance, we get:
    ray_distance: float = (
        (query_point[0] * ray_direction[0] + query_point[1] * ray_direction[1] + query_point[2] * ray_direction[2]
         - ray_source[0] * ray_direction[0] - ray_source[1] * ray_direction[1] - ray_source[2] * ray_direction[2])
        /
        ((ray_direction[0] ** 2) + (ray_direction[1] ** 2) + (ray_direction[2] ** 2)))

    if ray_distance < 0 and forward_only:
        return ray_source  # point is behind the source, so the closest point is just the source

    ray_point = [0.0] * 3  # temporary values
    for i in range(0, 3):
        ray_point[i] = ray_source[i] + ray_distance * ray_direction[i]
    return ray_point

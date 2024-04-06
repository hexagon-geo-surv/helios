import numpy as np
import matplotlib.pyplot as plt

seed_value = 42
rng = np.random.default_rng(seed=seed_value)


def generate_points_with_min_distance(density, grid_size_x, grid_size_y, min_dist):
    """
    Function to generate randomly distributed points given a point density and a minimum distance between points
    Starts with an equidistant grid of points and then perturbing the points but keeping the distance between the points at most `min_dist`
    source: https://stackoverflow.com/questions/27499139/how-can-i-set-a-minimum-distance-constraint-for-generating-points-with-numpy-ran
    """

    # number of points from density
    n = int(grid_size_x * grid_size_y * density)

    # compute grid shape based on number of points
    width_ratio = grid_size_y / grid_size_x
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced points
    x = np.linspace(0., grid_size_y - 1, num_x, dtype=np.float32)
    y = np.linspace(0., grid_size_x - 1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # plot for validation
    # plt.scatter(coords[:, 0], coords[:, 1], c="orange", s=1, alpha=0.5)

    # compute spacing
    init_dist = np.min((x[1] - x[0], y[1] - y[0]))

    # perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = rng.uniform(low=-max_movement,
                        high=max_movement,
                        size=(len(coords), 2))
    coords += noise

    # plot for validation
    plt.scatter(coords[:, 0], coords[:, 1], c="green", s=1)
    plt.show()

    return coords


def sampling_grid(grid_size_x, grid_size_y, grid_spacing, tree_positions, min_distance_to_trees):
    """
    Function to generate a regular sampling grid for TLS
    """
    # number of points in each direction
    num_x = np.int32(np.ceil(grid_size_x / grid_spacing))
    num_y = np.int32(np.ceil(grid_size_y / grid_spacing))

    # create regularly spaced positions
    x = np.linspace(0., grid_size_y - 1, num_x, dtype=np.float32)
    y = np.linspace(0., grid_size_x - 1, num_y, dtype=np.float32)
    scan_pos = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    return scan_pos


def create_scanner_setting_list(pulse_freq,
                                vertical_fov,
                                horizontal_res,
                                vertical_res,
                                scan_angle,
                                # all parameters that helios.ScanSettings take
                                combination_mode="zip"):
    """
    Function which creates a list of scann settings for combinations of settings provided, i.e.,
    when a list instead of a single value is provided for some of the paramters, they are combined
    in multiple scan settings (either via "zip" or as "cartesian product")

    returns a list of helios.ScanSettings with the different combinations
    """
    pass


def create_platform_setting_list(speed,
                                 altitude,
                                 combination_mode="zip"):
    """
    Same as above, but for platforms
    """
    pass

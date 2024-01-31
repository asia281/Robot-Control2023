import numpy as np

def are_points_on_same_plane(point_set):
    x1 = point_set[:, 0]
    y1 = point_set[:, 1]
    x2 = point_set[:, 2]
    y2 = point_set[:, 3]

    A = np.array([
        [x1[0], y1[0], 1],
        [x1[1], y1[1], 1],
        [x1[2], y1[2], 1]
    ])

    B = np.array([
        [x2[0]],
        [x2[1]],
        [x2[2]]
    ])

    coefficients = np.linalg.solve(A, B)

    residuals = np.abs(np.dot(A, coefficients) - B)
    indices_on_plane = np.where(residuals < 1e-5)[0]

    return indices_on_plane

def main():
    data = np.loadtxt(stdin)

    indices_on_plane = are_points_on_same_plane(data)

    print("Indices of points on the same plane:", indices_on_plane)

if __name__ == "__main__":
    from sys import stdin
    main()

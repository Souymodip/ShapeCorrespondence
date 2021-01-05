import Art
import ArtCollection
import numpy as np


def angle(segment1, segment2):
    v1 = segment1[1] - segment1[0]
    v2 = segment2[1] - segment2[0]

    if np.linalg.norm(v2) < 1.e-4 or np.linalg.norm(v1) < 1.e-4:
        return 0
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


def inside(art, point):
    vertices = art.get_vertices()
    start_line = np.array([point, vertices[0]])
    theta = 0
    for i in range(1, len(vertices)):
        next_line = np.array([point, vertices[i]])
        theta = theta + angle(start_line, next_line)
        start_line = next_line

    def divisible(v, d):
        t = v / d
        return (t - np.floor(t)) <= 1.e-5 or (np.ceil(t) - t) <= 1.e-5

    return divisible(theta, 2 * np.pi)





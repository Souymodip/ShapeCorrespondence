import Art
import numpy as np
import testsLevel1
import copy


'''
Smoothness: The max/min function is aproximated with smooth max/min. The smoothness parameter controls the level
of approximation. Higher value of smoothness implies greater accuracy
'''
SMOOTHNESS = 7
# import matplotlib.pyplot as plt


def frechet_distance(points1, points2):
    distance_mat = np.zeros(shape=(len(points1), len(points2)))
    for ij in np.ndindex(distance_mat.shape):
        i, j = ij
        distance_mat[ij] = np.linalg.norm(points1[i] - points2[j])

    distance, match = 0.0, []
    for i in np.ndindex(distance_mat.shape[:1]):
        j = np.argmin(distance_mat[i])
        match.append((i, j))
        distance = distance + distance_mat[i,j]
    return distance, match


def rotate_points(points, radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[c * p[0] - s * p[1], s * p[0] + c * p[1]] for p in points])


def smooth_min(x, smoothness):
    assert (smoothness > 0)
    maximum = np.max(x)
    if maximum < 1.e-7:
        return 0
    y = np.log(np.sum(np.exp(-smoothness * x)))/(-smoothness)
    return y


def derivative_smooth_min(x, smoothness, dx): # soft min
    alpha = -smoothness
    d = np.sum(np.exp(alpha * x))
    if dx.shape != x.shape:
        print("x:{}, dx:{}".format(x.shape, dx.shape))
        assert (0)
    return np.sum(np.exp(alpha * x) * dx * (1/d))


def differentiable_frechet_distance(points1, points2):
    def distance(point, points):
        l = np.array([np.linalg.norm(point - q) for q in points])
        return smooth_min(l, smoothness=SMOOTHNESS)

    return np.sum([distance(p, points2) for p in points1])


def gradient_descent(x, cost, gradient_update, error, max_step):
    curr_x = x
    theta = 0
    steps = 0
    total_change = theta
    current_cost = cost(curr_x)
    next_x, new_theta = gradient_update(curr_x)
    next_cost = cost(next_x)

    print("{}. Current Cost: {}, Next Cost: {}".format(steps, current_cost, next_cost))
    # print("\tCurrent x:{}, \n\tNext x   :{}".format(curr_x, next_x))


    while np.abs(np.abs(current_cost) - np.abs(next_cost)) > error and steps < max_step:
        #update
        steps = steps + 1
        current_cost = next_cost
        curr_x = next_x
        theta = new_theta
        total_change = total_change + theta

        next_x, new_theta = gradient_update(curr_x)
        next_cost = cost(next_x)
        print("{}. Current Cost: {}, Next Cost: {}".format(steps, current_cost, next_cost))
        # print("\tCurrent x:{}, \n\tNext x   :{}".format(curr_x, next_x))

        if np.sign(theta) + np.sign(new_theta) == 0:
            break

    return next_cost, total_change


def get_best_fit(t1, index1, t2, index2):
    art1, art2 = copy.deepcopy(t1), copy.deepcopy(t2)
    p1, p2 = art1.get_vertex(index1), art2.get_vertex(index2)
    art1.apply(Art.Translate(-p1))
    art2.apply(Art.Translate(-p2))
    points1, points2 = np.roll(art1.get_vertices(), -index1), np.roll(art2.get_vertices(), -index2)

    learning_rate = 0.01

    def cost(x): # x = current points1
        return differentiable_frechet_distance(x, points2)

    # Rotational derivative
    def gradient_update(x):
        def dx(curr_x_i): # curr_x_i = current points1
            ret = []
            for p2 in points2:
                l = np.linalg.norm(curr_x_i - p2)
                ret.append((p2[0] * curr_x_i[1] - p2[1] * curr_x_i[0]) / np.linalg.norm(curr_x_i - p2) if l > 1.e-7 else 0)
            return np.array(ret)

        def gradient(curr_x): # curr_x = current points1
            sum_x = 0
            for x_i in curr_x:
                vec = np.linalg.norm(x_i - points2, axis=1)
                v_dx = dx(x_i)
                sum_x = sum_x + derivative_smooth_min(vec, smoothness=SMOOTHNESS, dx=v_dx)
            return sum_x

        grad = gradient(curr_x=x)
        print("\t Current Gradient: {}".format(grad))
        radian = - learning_rate * grad
        new_x = rotate_points(x, radian)
        # print("\t x:{} new_x:{}".format(x, new_x))
        return new_x, radian

    return gradient_descent(points1, cost=cost, gradient_update=gradient_update, error=0.1, max_step=50)


def best_align(art1, art2):
    cost, radian = np.inf, 0
    for i in range(art2.no_of_vertices()):
        i_cost, i_radian = get_best_fit(art1, 0, art2, i)
        if i_cost < np.inf:
            radian = i_radian

    return radian



def main():
    art1 = Art.PieceWiseBezier(np.array([
        [[3.37866, -4.10971], [2.29696, -5.19142], [4.46037, -3.028]],
        [[7.08922, -2.65872], [5.69301, -2.28545], [8.48543, -3.03199]],
        [[11.5197, -4.64137], [9.49883, -4.9747], [13.5406, -4.30805]],
        [[15.3743, -2.02737], [13.7746, -2.16383], [16.974, -1.89091]]
    ]), is_closed=True, show_control=False)
    art2 = Art.PieceWiseBezier(np.array([
        [[3.37866, -4.10971], [2.29696, -5.19142], [4.46037, -3.028]],
        [[7.08922, -2.65872], [5.69301, -2.28545], [8.48543, -3.03199]],
        [[11.5197, -4.64137], [9.49883, -4.9747], [13.5406, -4.30805]],
        [[15.3743, -2.02737], [13.7746, -2.16383], [16.974, -1.89091]]
    ]), is_closed=True, show_control=False)
    art2.apply(Art.Rotate(30, art1.get_vertex(index=0)))
    art3 = copy.deepcopy(art2)

    least_cost, change = get_best_fit(t1=art1, index1=0, t2=art2, index2=0)
    art3.apply(Art.Rotate(-change, art3.get_vertex(index=0)))
    art3.set_color((0,0, 255))

    print("-------------------- \n Cost : {}".format(least_cost))

    d = Art.Draw()
    d.add_art(art1)
    d.add_art(art2)
    d.add_art(art3)
    d.draw()


if __name__ == '__main__':
    main()


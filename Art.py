import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math
import Pixel
import numpy as np
import copy

WIDTH = 12
HEIGHT = 10
SCALE = 2


def get_code(rgb=(255,255,255)):
    code = str(hex((rgb[0]*65536)+(rgb[1]*256)+rgb[2]))[2:]
    new_code = code
    for i in range(6-len(code)):
        new_code = '0' + new_code
    return "#"+new_code


def print_raster(raster):
    s = ""
    for ij in np.ndindex(raster.shape[:2]):
        if ij[1] == 0:
            s = s + "\n"
        else:
            s = s + "{:.3f}".format(raster[ij][0]) + ", "

    print(s)


def rotate_points(points, radian, about):
    c = np.cos(radian)
    s = np.sin(radian)
    dp = points - about
    return np.array([[c * p[0] - s * p[1] + about[0], s * p[0] + c * p[1] + about[1]] for p in dp])


def quad_roots(a, b, c):
    if a == 0:
        assert (b!=0)
        return np.array([-c/b])
    disc = b**2 - 4*a*c
    if disc > 0:
        sq_disc = np.sqrt(disc)
        return np.array([(-b + sq_disc)/(2*a), (-b - sq_disc)/(2*a)])
    else:
        return np.array([])


class Art:
    def __init__(self):
        self.point_buffer = ([], [])
        self.color = '#000000'
        self.alpha = 1
        self.fill_color = '#FFFFFF'
        self.toFill = False
        self.diffuse = False

    def set_point_buffer(self, xs, ys):
        self.point_buffer = xs, ys

    def set_color(self, rgb):
        self.color = get_code(rgb)

    def get_point_buffer(self):
        return self.point_buffer

    def set_fill_color(self, rgb):
        self.fill_color = get_code(rgb)
        self.toFill = True

    def set_fill_color_alpha(self, rgb, a):
        self.set_fill_color(rgb)
        self.set_alpha(a)

    def get_fill_color(self):
        return self.fill_color

    def set_alpha(self, a):
        self.alpha = a

    def get_alpha(self):
        return  self.alpha

    def reset_fill(self):
        self.toFill = False

    def set_diffuse(self, v=True):
        self.diffuse = True

    def add(self, ax):
        if self.toFill:
            ax.fill(self.point_buffer[0], self.point_buffer[1], color=self.fill_color, alpha=self.alpha)
        else:
            ax.plot(self.point_buffer[0], self.point_buffer[1], color=self.color, alpha=self.alpha)


class Rectangle(Art):
    def __init__(self, top_left, bottom_right):
        super().__init__()
        super().set_point_buffer([top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]], \
                               [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]])

    def get_bottom_left(self):
        if len(self.point_buffer[0]) == 5 :
            return [self.point_buffer[0][0], self.point_buffer[1][0]]
        else:
            return None

    def get_top_right(self):
        if len(self.point_buffer[0]) == 5 :
            return [self.point_buffer[0][2], self.point_buffer[1][2]]
        else:
            return None

    def get_right(self):
        if len(self.point_buffer[0]) == 5 :
            return self.point_buffer[0][2]
        else:
            return None

    def get_left(self):
        if len(self.point_buffer[0]) == 5 :
            return self.point_buffer[0][0]
        else:
            return None

    def get_bottom(self):
        if len(self.point_buffer[0]) == 5 :
            return self.point_buffer[1][0]
        else:
            return None

    def get_top(self):
        if len(self.point_buffer[0]) == 5 :
            return self.point_buffer[1][2]
        else:
            return None

    def is_inside(self, point):
        return (self.get_left() <= point[0] <= self.get_right()) or (self.get_bottom() <= point[0] <= self.get_top())


class Line(Art):
    def __init__(self, start, end):
        super().__init__()
        super().set_point_buffer([start[0], end[0]], [start[1], end[1]])

    def add_diffuse(self, ax):
        x0, y0 = self.point_buffer[0][0], self.point_buffer[1][0]
        x1, y1 = self.point_buffer[0][1], self.point_buffer[1][1]
        xp = x0
        yp = y0
        DIV = 50
        ax.plot([x0,x1],[y0,y1], color=self.color, alpha=self.alpha, linewidth=0.1)
        for t in range(1, DIV):
            x = (x0 + t/DIV*(x1-x0))
            y = (y0 + t/DIV*(y1-y0))
            x3 = (x0 + (1 - t/DIV)*(x1-x0))
            y3 = (y0 + (1 - t/DIV)*(y1-y0))
            width = t if t < 4 else ((DIV-t) if t > DIV - 4 else 4)
            ax.plot([x,x3],[y,y3], color=self.color, alpha=self.alpha*0.01, linewidth=width)

    def add(self, ax):
        if self.diffuse:
            self.add_diffuse(ax)
        else:
            super().add(ax)

class Circle(Art):
    def __init__(self, center, radius):
        super().__init__()
        self.center = center
        self.radius = radius

    def add(self,ax):
        if self.toFill:
            c = plt.Circle((self.center[0], self.center[1]), radius=self.radius, color=self.fill_color, alpha=self.alpha)
        else:
            c = plt.Circle((self.center[0], self.center[1]), radius=self.radius, edgecolor=self.color, alpha=self.alpha, fill=False)
        ax.add_artist(c)


class Bezier(Art):
    def __init__(self, controls, show_control=False):
        super().__init__()
        self.controls = controls
        self.show_controls = show_control
        self.DISCRETE = 10
        assert(self.DISCRETE > 0)

    def apply(self, T):
        self.controls = T.apply(self.controls)
        return self

    def set_color(self, rgb):
        self.color = get_code(rgb)

    def point_at(self, t):
        p = self.controls
        return p[0] * ((1 - t) ** 3) + 3 * p[1] * ((1 - t) ** 2) * t + 3 * p[2] * (1 - t) * (t ** 2) + p[3] * (t ** 3)

    def aligned(self):
        p = copy.deepcopy(self.controls) - self.controls[0]
        if p[-1][1] != p[0][1]:
            radian = np.pi / 2 if p[0][0] == p[-1][0] else \
                np.arctan((p[-1][1] - p[0][1]) / (p[-1][0] - p[0][0]))
            p = rotate_points(points=p, radian=-radian, about=p[0])
        return Bezier(p)

    def get_extremes(self):
        p = self.aligned().controls

        a = (3*(-p[0] + 3*p[1] -3*p[2] + p[3]))[1]
        b = (6*(p[0] - 2*p[1] + p[2]))[1]
        c = (3*(p[1] - p[0]))[1]

        if a == b == 0:
            r = np.array([0]) if p[0][1] > p[3][1] else np.array([1])
        else:
            r = quad_roots(a, b, c)
            r = np.array([t for t in r if 0 <=t <= 1])
            r = np.sort(r)
            if r.size == 0:
                r = np.array([0]) if p[0][1] > p[3][1] else np.array([1])

        return np.array([self.point_at(t) for t in r])

    def add(self, ax):
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        if len(self.controls) == 4:
            path = Path(self.controls, codes)
            p = patches.PathPatch(path, facecolor='none', lw=2, edgecolor=self.color)
            ax.add_patch(p)
            if self.show_controls:
                xs, ys = zip(*self.controls)
                ax.plot(xs, ys, '--o', color='red')

    def length(self):
        assert (len(self.controls) == 4)
        length = 0

        def split(controls, t):
            p0 = controls[0]
            p3 = controls[3]
            p01 = t * p0 + (1 - t) * controls[1]
            p12 = t * controls[1] + (1 - t) * controls[2]
            p23 = t * controls[2] + (1 - t) * p3
            p0112 = t * p01 + (1 - t) * p12
            p1223 = t * p12 + (1 - t) * p23
            q = t * p0112 + (1 - t) * p1223
            return np.array([p0, p01, p0112, q]), np.array([q, p1223, p23, p3])

        def discrete_len(controls):
            c = np.linalg.norm(controls[0] - controls[-1])
            cc = np.linalg.norm(controls[0] - controls[1]) + \
                 np.linalg.norm(controls[1] - controls[2]) + \
                 np.linalg.norm(controls[2] - controls[3])
            return (c + cc) / 2
        t = 0
        p = self.controls
        for i in range(self.DISCRETE):
            t = t + 1/self.DISCRETE
            l, r = split(controls=p, t=t)
            length = length + discrete_len(l)
            p = r
        return length

    def get_point(self, t):
        assert (0<= t <= 1)
        p0 = self.controls[0]
        p1 = self.controls[1]
        p2 = self.controls[2]
        p3 = self.controls[3]
        return ((1-t) ** 3) * p0 + 3 * t * ((1-t)**2) * p1 + 3 * (t**2) * (1-t) * p2 + (t**3) * p3

    def get_gradient(self, t):
        assert (0 <= t <= 1)
        p0 = self.controls[0]
        p1 = self.controls[1]
        p2 = self.controls[2]
        p3 = self.controls[3]
        return 3 * ((1-t)**2) * (p1 - p0) + 6 * (1-t) * t * (p2 - p1) + 3 * (t**2) * (p3 - p2)


class PieceWiseBezier(Art):
    def __int__(self, beziers):
        super().__init__()
        self.beziers = beziers

    def __init__(self, anchors, is_closed=True, show_control=False):
        super().__init__()
        assert(anchors.shape[1] == 3 and anchors.shape[2] == 2)
        self.is_closed = is_closed
        self.beziers = []
        for i in range(anchors.shape[0] - 1):
            self.beziers.append(Bezier([
                anchors[i][0],
                anchors[i][2],
                anchors[i+1][1],
                anchors[i+1][0]
            ], show_control))
        if is_closed:
            self.beziers.append(Bezier([
                anchors[-1][0],
                anchors[-1][2],
                anchors[0][1],
                anchors[0][0]
            ], show_control))
        self.show_controls = show_control

    def get_beziers(self):
        return self.beziers

    def set_color(self, rgb):
        for b in self.beziers:
            b.set_color(rgb)

    def apply(self, T):
        self.beziers = [b.apply(T) for b in self.beziers]
        return self

    def add(self, ax):
        for b in self.beziers:
            b.add(ax)

    def no_of_vertices(self):
        return len(self.beziers) if self.is_closed else len(self.beziers) + 1

    def get_vertices(self):
        v = np.array([b.controls[0] for b in self.beziers])
        return np.append(v, [self.beziers[-1].controls[3]], axis=0)

    def get_vertex(self, index):
        assert (index <= len(self.beziers))
        return self.beziers[index].controls[0] if index < len(self.beziers) else self.beziers[-1].controls[3]

    def get_centroid(self):
        vertices = self.get_vertices()
        return np.mean(vertices, axis=0)

    def split_bezier_in_parts(self, index, parts):
        while parts > 1:
            r = 1 / parts
            self.split_bezier(index, r)
            index = index + 1
            parts = parts - 1

    def no_of_beziers(self):
        return len(self.beziers)

    def get_bezier(self, index):
        return self.beziers[index%len(self.beziers)]

    def get_bounding_box(self):
        points = []
        for b in self.beziers:
            for i in range(3):
                points.append(b.controls[i])
        points = np.array(points)
        x, y = np.hsplit(points, 2)
        return np.array([[np.min(x), np.min(y)], [np.max(x), np.max(y)]])

    def split_bezier(self, index, t):
        assert (0 < t < 1)
        b = self.beziers[index]
        p0 = b.controls[0]
        p3 = b.controls[3]
        p01 = t * p0 + (1-t) * b.controls[1]
        p12 = t * b.controls[1] + (1-t) * b.controls[2]
        p23 = t * b.controls[2] + (1-t) * p3
        p0112 = t * p01 + (1-t) * p12
        p1223 = t * p12 + (1-t) * p23
        q = t * p0112 + (1-t) * p1223

        self.beziers[index] = Bezier([
            p0, p01, p0112, q
        ], self.show_controls)

        self.beziers.insert(index+1, Bezier([
            q, p1223, p23, p3
        ], self.show_controls))

    def increase_by(self, factor):
        assert (factor >= 2)
        for i in range(len(self.beziers)):
            self.split_bezier_in_parts(i, factor)


class Polygon(Art):
    def __init__(self, points, isClosed=True):
        super().__init__()
        self.points = points
        self.isClosed = isClosed

    def apply(self, T):
        self.points = T.apply(self.points)

    def copy(self):
        return Polygon(self.points, self.isClosed)

    def add(self, ax):
        if len(self.points) > 0 :
            x0 = self.points[0][0]
            y0 = self.points[0][1]

            for i in range (1, len(self.points)):
                x1 = self.points[i][0]
                y1 = self.points[i][1]
                ax.plot([x0, x1], [y0, y1], color=self.color)
                x0 = x1
                y0 = y1

            if self.isClosed:
                l = len(self.points)
                x0 = self.points[l-1][0]
                y0 = self.points[l-1][1]
                x1 = self.points[0][0]
                y1 = self.points[0][1]
                ax.plot([x0, x1], [y0, y1], color=self.color)

    def area(self, quary_region):
        return Pixel.area(polygon=self, qr=quary_region)

    def grad_area(self, qr):
        return Pixel.grad_area(self, qr)

    def get_centroid(self):
        return np.mean(self.points, axis=0)


class Function(Art):
    def __init__(self, fx, clip_rect = None):
        super().__init__()
        self.Fx = fx
        self.clip = clip_rect if clip_rect else Rectangle([0,0], [WIDTH*100, HEIGHT*100])

    def draw_in_range(self,ax, left, right, alpha = None):
        x = []
        y = []
        for i in range(int(left), int(right)):
            fx = self.Fx(float(i))
            if self.clip.get_bottom_left()[1] <= fx <= self.clip.get_top_right()[1]:
                y.append(fx)
                x.append(i)
        ax.plot(x, y, color=self.color, alpha=alpha if alpha else self.alpha)
        return x, y

    def add(self, ax):
        if self.diffuse:
            self.add_diffuse(ax)
        else:
            self.draw_in_range(ax, self.clip.get_bottom_left()[0], self.clip.get_top_right()[0])

    def add_diffuse(self, ax):
        x, y = self.draw_in_range(ax, self.clip.get_bottom_left()[0], self.clip.get_top_right()[0], 0.01)
        DIV = len(x)
        W = 4
        R = 1
        width = 1
        a = 0
        for t in range(1, int(DIV/2)):
            a = a + self.alpha * 0.01
            if a <= self.alpha:
                if width < W:
                    width = width + t / R
                ax.plot(x[t:DIV-t], y[t:DIV-t], color=self.color, alpha=self.alpha * 0.01, linewidth=width)
            else:
                break


# class Transform:
#     def __init__(self):

class Translate:
    def __init__(self, delta):
        self.delta = delta

    def apply(self, points):
        return [c + self.delta for c in points]


class FlipX:
    def __init__(self, x = 0):
        self.x = x

    def apply(self, points):
        return [np.array([-c[0] + 2*self.x , c[1]]) for c in points]


class Scale:
    def __init__(self, scale):
        self.scale = scale

    def apply(self, points):
        return [self.scale * c for c in points]

class Rotate:
    def __init__(self, radian, origin = (0,0)):
        self.delta = radian
        self.origin = origin

    def apply(self, points):
        c = math.cos(self.delta)
        s = math.sin(self.delta)

        def rotate_point(p):
            dp = p - self.origin
            return np.array([c * dp[0] - s * dp[1] + self.origin[0], s * dp[0] + c * dp[1] + self.origin[1]])

        return [rotate_point(p) for p in points]


class Draw:
    def __init__(self, height=HEIGHT, width=WIDTH, scale=SCALE):
        self.HEIGHT = height
        self.WIDTH = width
        self.SCALE = scale
        self.art_buffer = []
        self.rasters = []

    def flush(self):
        self.art_buffer.clear()
        self.rasters.clear()

    def add_art(self, art):
        self.art_buffer.append(art)

    def add_raster(self, anp, bot_left = (0, 0), scale=1):
        assert(len(anp.shape) == 3)
        self.rasters.append((anp, bot_left, scale))

    def draw (self):
        fig = plt.figure(figsize=(self.WIDTH, self.HEIGHT))
        ax = fig.add_subplot()
        plt.xlim(-self.SCALE*self.WIDTH, self.SCALE*self.WIDTH)
        plt.ylim(-self.SCALE*self.HEIGHT, self.SCALE*self.HEIGHT)
        plt.autoscale(False)
        boundary = [-SCALE*self.WIDTH, SCALE*self.WIDTH], [SCALE*self.HEIGHT, SCALE*self.HEIGHT]
        ax.plot(boundary[0],boundary[1], color='#000000', alpha=0)
        for art in self.art_buffer:
            art.add(ax)

        for r, a, s in self.rasters:
            scx = np.zeros((r.shape[0] * r.shape[1]))
            scy = np.zeros((r.shape[0] * r.shape[1]))
            sc = np.zeros((r.shape[0] * r.shape[1]))
            count = 0
            for ij in np.ndindex(r.shape[:2]):
                qx, qy = ij[0] / s + a[0], ij[1] / s + a[1]
                scx[count] = qx
                scy[count] = qy
                sc[count] = np.abs(r[ij])
                count = count + 1

            ax.scatter(scx, scy, s=5/s, c=sc, cmap="Blues", edgecolors=None)
            # plt.axis("off")
            print("Raster Intensity := {:.5f}".format(np.sum(r)))
        plt.show()


def draw_match(art1, art2, m, draw, window=1):
    points1, points2 = art1.get_vertices(), art2.get_vertices()
    for ij in m:
        i,j = ij
        c1 = Circle(points1[i], 0.2)
        c2 = Circle(points2[j], 0.2)
        c1.set_color((255,0 , 0))
        c2.set_color((0,0, 255))
        l = Line(points1[i], points2[j])
        l.set_color((0,255,0))
        draw.add_art(l)
        draw.add_art(c1)
        draw.add_art(c2)
        bs = [art1.get_bezier(k) for k in range(i, i + window)]
        for b in bs:
            b.set_color((255, 255, 0))
            draw.add_art(b)

        bs = [art2.get_bezier(k) for k in range(j, j + window)]
        for b in bs:
            b.set_color((0, 255, 255))
            draw.add_art(b)
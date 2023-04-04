import numpy as np
import matplotlib.pyplot as plt


def subtended_angle(hull, point):
    hull_x_coord = hull[:, 0]
    hull_y_coord = hull[:, 1]

    Nvert = hull.shape[0]
    Nedge = Nvert - 1

    Totang = 0
    for i in range(Nedge):
        v1 = point - hull[i]
        v2 = point - hull[i+1]
        ang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        ang = np.rad2deg(np.arccos(ang))
        Totang += ang

    return Totang

# N = 4
# hull_points = [[4*np.cos(2*np.pi*k/N), 4*np.sin(2*np.pi*k/N)] for k in range(N)]
# hull_points.append(hull_points[0])
# hull_points = np.array(hull_points)

hull_points = [[2, 3], [1, 3], [0, 0.5], [0, -0.5], [1, -2], [2, -2], [2, 3]]
hull_points = np.array(hull_points)

hull_x_coord = hull_points[:, 0]
hull_y_coord = hull_points[:, 1]

plt.grid(True)
plt.plot(hull_x_coord, hull_y_coord, 'bo-', lw=3)

ck_pts = [[0, 0]]
ck_pts = np.array(ck_pts)
x_coord = ck_pts[:, 0]
y_coord = ck_pts[:, 1]
plt.plot(x_coord, y_coord, 'ro')

for i in range(ck_pts.shape[0]):
    print(subtended_angle(hull_points, ck_pts[i]))

plt.show()

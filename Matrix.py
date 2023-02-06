import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation(t, u):
    return np.matrix([[np.cos(t) + u[0]**2 * (1 - np.cos(t)), u[0]*u[1]*(1 - np.cos(t)) - u[2]*np.sin(t), u[0]*u[2]*(1 - np.cos(t)) + u[1]*np.sin(t)],
                    [u[1]*u[0]*(1 - np.cos(t)) + u[2]*np.sin(t), np.cos(t) + u[1]**2 * (1 - np.cos(t)), u[1]*u[2]*(1 - np.cos(t))-u[0]*np.sin(t)],
                    [u[2]*u[0]*(1 - np.cos(t)) - u[1]*np.sin(t), u[2]*u[1]*(1 - np.cos(t)) + u[0]*np.sin(t), np.cos(t) + u[2]**2 * (1 - np.cos(t))]])

#SOURCE: https://www.wikiwand.com/en/Rotation_matrix#Rotation_matrix_from_axis_and_angle


def mirror_normal(input, reflected):

    sun = input/np.linalg.norm(input)
    # direction of Sun, normalized

    receiver = reflected/np.linalg.norm(reflected)
    # direction of receiver, normalized

    rotation_axis = np.cross(receiver, sun)/np.linalg.norm(np.cross(receiver, sun))
    # takes the cross product of the incoming and reflected ray, this defines the rotation axis

    magnitude = np.dot(receiver, sun)
    # as the directionvectprs are normalized, this is the cosine of the inbetween angle

    angle = np.arccos(magnitude)/2
    # we need to divide the angle by 2 (law of reflections)

    mirror_normal = np.array(rotation(angle, rotation_axis)@receiver)[0]
    # finally, the mirror normal is the reflected ray rotated by the angle around the rot. axis

    print('ROTATION AXIS=' ,rotation_axis)
    print('SUN DIRECTION=',sun)
    print('RECEIVER DIRECTION=' ,receiver)
    print('ROTATION ANGLE=',angle)
    print('MIRROR NORMAL=',mirror_normal)

    soa = np.array([[0, 0, 0, sun[0], sun[1], sun[2]], [0, 0, 0, receiver[0], receiver[1], receiver[2]],
                    [0, 0, 0, mirror_normal[0], mirror_normal[1], mirror_normal[2]], [0, 0, 0, rotation_axis[0], rotation_axis[1], rotation_axis[2]]])

    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show()

input = np.array([-1, -7, 1])
reflected = np.array([1, -4, 1])

mirror_normal(input, reflected)
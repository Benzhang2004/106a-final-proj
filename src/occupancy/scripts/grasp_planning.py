#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Starter code for EE106B grasp planning project.
Author: Amay Saxena, Tiffany Cappellari
Modified by: Kirthi Kumar
"""
# may need more imports
import numpy as np
from utils import *
import trimesh
import vedo
from trimesh.sample import sample_surface_even
from trimesh.proximity import closest_point

MAX_GRIPPER_DIST = 0.075
MIN_GRIPPER_DIST = 0.022
GRIPPER_LENGTH = 0.105

ERROR_TOL = 1e-3 # for checking cvxpy solutions
GRAVITATIONAL_CONSTANT = 9.8
NOISE_STD = 0.04 # for robust force closure
NUM_TRIALS_ROBUST = 500 # for robust force closure
NUM_TRIAlS_FER = 1000 # for Ferrari Canny
NUM_TRIALS_PLANNER = 20

RANDOM_SEED = 24107  # vary the seeds to get different grasps
                     # good seeds for nozzle: (24103,2) (24113,1) (24107,2)
                     # good (seed,sample amount) for pawn: (24113,2) (24103,2) (24114,2)
SAMPLE_AMOUNT = 20 # the number of vertex pairs for which we will evaluate a potential grasp
                  # sometimes, a SAMPLE_AMOUNT of samples cannot be returned. Adjust the number as necessary

METRIC_INDEX = 1 # determines which metric we are testing: gravity==0, robust==1, ferrari==2

CONTACT_BASIS = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]], dtype=np.float64) # for soft contact model

import cvxpy as cvx # suggested, but you may change your solver to anything you'd like (ex. casadi)



def find_intersections(mesh, p1, p2):
    ray_origin = (p1 + p2) / 2
    ray_length = np.linalg.norm(p1 - p2)
    ray_dir = (p2 - p1) / ray_length
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin, ray_origin],
        ray_directions=[ray_dir, -ray_dir],
        multiple_hits=True)
    dist_to_center = np.linalg.norm(locations - ray_origin, axis=1)
    dist_mask = dist_to_center <= (ray_length / 2) # only keep intersections on the segment.
    on_segment = locations[dist_mask]
    faces = index_tri[dist_mask]
    return on_segment, faces

def normal_at_point(mesh, p):
    point, dist, face = proximity.closest_point(mesh, [p])
    if dist > 0.001:
        print("Input point is not on the surface of the mesh!")
        return None
    return mesh.face_normals[face[0]]


def normalize(vec):
    return vec / np.linalg.norm(vec)

def hat(v):
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
    elif v.shape == (6, 1) or v.shape == (6,):
        return np.array([
                [0, -v[5], v[4], v[0]],
                [v[5], 0, -v[3], v[1]],
                [-v[4], v[3], 0, v[2]],
                [0, 0, 0, 0]
            ])
    else:
        raise ValueError

def adj(g):
    if g.shape != (4, 4):
        raise ValueError

    R = g[0:3,0:3]
    p = g[0:3,3]
    result = np.zeros((6, 6))
    result[0:3,0:3] = R
    result[0:3,3:6] = np.matmul(hat(p), R)
    result[3:6,3:6] = R
    return result

def look_at_general(origin, direction):
    up = np.array([0, 0, 1])
    z = normalize(direction) # create a z vector in the given direction
    x = normalize(np.cross(up, z)) # create a x vector perpendicular to z and up
    y = np.cross(z, x) # create a y vector perpendicular to z and x

    result = np.eye(4)

    # set rotation part of matrix
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z

    # set translation part of matrix to origin
    result[0:3,3] = origin

    return result


def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
    if mu <= 0 or gamma <= 0:
        return False
    # Instead of following the hint, I choose to use Proposition 5.2 in MLS
    # checking G is full-rank
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    if G is None:
        return False
    
    # FYI, here num_facets, mu, gamma are not used
    if np.linalg.matrix_rank(G) < 6:
        return False
    # checking the existence of interior force
    # only needs to consider force (not torque), only needs to check 
    cos_threshold = - (1 - mu**2) / (1 + mu**2) # tan(theta) = mu, -cos(2theta) is this
    cos_value = np.dot(normalize(normals[0]), normalize(normals[1])) # normalization might not be necessary
    if cos_value >= cos_threshold:
        return False
    return True
    # TODO: the problem here is we do not have any constraints on gamma
    # TODO: assume normal vectors are pointing outwards (or both pointing inwards)
    

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    # print("start get_grasp_map")
    # print(f"mat 1: {look_at_general(vertices[0], -normals[0])}")
    # print(f"mat 2: {look_at_general(vertices[1], -normals[1])}")
    G = np.zeros((6, 8))
    g1 = look_at_general(vertices[0], -normals[0])
    g2 = look_at_general(vertices[1], -normals[1])
    if np.isnan(g1).any() or np.isnan(g2).any():
        #print("unsuccesful get_grasp_map")
        return None 
    
    adj1 = adj(np.linalg.inv(look_at_general(vertices[0], -normals[0]))) # TODO: check the direction of normal vectors
    adj2 = adj(np.linalg.inv(look_at_general(vertices[1], -normals[1])))
    G[:, 0:4] = adj1.T @ CONTACT_BASIS
    G[:, 4:8] = adj2.T @ CONTACT_BASIS
    #print("succesful get_grasp_map")
    return G
    

def get_F_matrix(num_facets, mu):
    # wrench = G @ F @ alpha, in principle
    # f_1 through f_{num_facets} correspond to the discretized edges
    # where f_i = [cos(theta)*mu, sin(theta)*mu, 1, 0]
    # f_0 corresponds to the normal direction in z, i.e. [0, 0, 1, 0]
    # f_{num_facets + 1} corresponds to the torsion friction, i.e. [0, 0, 0, 1]
    F = np.zeros((4, num_facets + 2))
    F[:, 0] = np.array([0, 0, 1, 0])
    angles = np.arange(num_facets) * 2 * np.pi / num_facets
    F[0, 1:num_facets+1] = np.cos(angles) * mu
    F[1, 1:num_facets+1] = np.sin(angles) * mu
    F[2, 1:num_facets+1] = 1
    F[3, 1:num_facets+1] = 0
    F[:, num_facets+1] = np.array([0, 0, 0, 1])
    # then, f^{\perp} = a_0 + a_1 + .metric.. + a_{num_facets}
    return F


def contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench):
    # F MATRIX
    F = get_F_matrix(num_facets, mu)

    # DEFINE CONTACT FORCES
    #    these are analogous to the alphas in the discretized friction cone formulas
    alpha = cvx.Variable(2 * (num_facets + 2))
    alpha_1, alpha_2 = alpha[:num_facets + 2], alpha[num_facets + 2:]
    
    # DEFINE CONSTRAINTS
    constraints = [
        alpha_1[:-1] >= 0,  # (alpha >= 0)
        alpha_2[:-1] >= 0,  # (alpha >= 0)
        gamma * cvx.sum(alpha_1[:-1]) >= alpha_1[-1], # (f_4 <= gamma f_3)
        gamma * cvx.sum(alpha_1[:-1]) >= -alpha_1[-1], # (-f_4 <= gamma f_3) 
        gamma * cvx.sum(alpha_2[:-1]) >= alpha_2[-1],
        gamma * cvx.sum(alpha_2[:-1]) >= -alpha_2[-1]
    ]
    
    # DEFINE THE PROBLEM 
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    if G is None:
        return False
    
    zeros = np.zeros((4, num_facets + 2))
    combined_F = np.vstack((np.hstack((F, zeros)), np.hstack((zeros, F)))) # (8, 2 * (num_facets + 2))
    cost = cvx.sum_squares(G @ combined_F @ alpha - desired_wrench)
    objective = cvx.Minimize(cost)
    problem = cvx.Problem(objective, constraints)
    
    try:
        problem.solve()
    except Exception as e:
        return False # Silent mode

    if problem.status == 'optimal' and np.isclose(problem.value, 0, atol=ERROR_TOL):
        return True
    else:
        return False
   

def _opt_min_force(vertices, normals, num_facets, mu, gamma, desired_wrench):
    # F MATRIX
    F = get_F_matrix(num_facets, mu)

    # DEFINE CONTACT FORCES
    #    these are analogous to the alphas in the discretized friction cone formulas
    alpha = cvx.Variable(2 * (num_facets + 2))
    alpha_1, alpha_2 = alpha[:num_facets + 2], alpha[num_facets + 2:]
    
    # DEFINE CONSTRAINTS
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    if G is None:
        return False
    
    zeros = np.zeros((4, num_facets + 2))
    combined_F = np.vstack((np.hstack((F, zeros)), np.hstack((zeros, F)))) # (8, 2 * (num_facets + 2))
    constraints = [
        alpha_1[:-1] >= 0,  # (alpha >= 0)
        alpha_2[:-1] >= 0,
        gamma * cvx.sum(alpha_1[:-1]) >= alpha_1[-1], # (f_4 <= gamma f_3)
        gamma * cvx.sum(alpha_1[:-1]) >= -alpha_1[-1], # (-f_4 <= gamma f_3) 
        gamma * cvx.sum(alpha_2[:-1]) >= alpha_2[-1],
        gamma * cvx.sum(alpha_2[:-1]) >= -alpha_2[-1],
        G @ combined_F @ alpha == desired_wrench # wrench constraint
    ]

    # DEFINE THE PROBLEM
    cost = cvx.norm(combined_F @ alpha) # note that we are miniminzing the l2 norm, so square the value!
    objective = cvx.Minimize(cost)
    problem = cvx.Problem(objective, constraints)

    problem.solve()
    return problem, G, combined_F @ alpha.value


def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass):
    gravity_wrench = np.array([0, 0, -GRAVITATIONAL_CONSTANT*object_mass, 0, 0, 0]) # TODO: check the negative sign
    try:
        problem, G, f = _opt_min_force(vertices, normals, num_facets, mu, gamma, -gravity_wrench) # TODO: check the negative sign
    except Exception as e:
        return np.inf # Silent mode
    
    if problem.status == 'optimal' and np.isclose(np.linalg.norm(G @ f + gravity_wrench), 0, atol=ERROR_TOL):
        return f[2] + f[6]
    else:
        return np.inf
    # the lower the better, ranges from (0, np.inf)


def sample_around_vertices(rng, delta, vertices, object_mesh):
    if vertices.shape == (3,):
        vertices = vertices.reshape((1, 3))
    assert len(vertices.shape) == 2 and vertices.shape[1] == 3 # (n, 3)
    vertices_hat = np.zeros_like(vertices)
    for n in range(vertices.shape[0]):
        normal = normal_at_point(object_mesh, vertices[n, :])
        R = look_at_general(np.zeros(3), normal)[0:3, 0:2] # [x, y]
        vertices_hat[n] = vertices[n] + R @ rng.normal(scale=delta, size=(2,))
    vertices_hat = closest_point(object_mesh, vertices_hat)[0]
    if vertices_hat.shape == (1, 3):
        vertices_hat = vertices_hat.reshape((3,))
    return vertices_hat


def compute_robust_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    success_count = 0
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(NUM_TRIALS_ROBUST):
        vertices_hat = sample_around_vertices(rng, NOISE_STD, vertices, mesh)
        norm1_hat = normal_at_point(mesh, vertices_hat[0, :])
        norm2_hat = normal_at_point(mesh, vertices_hat[1, :])
        normals_hat = np.vstack((norm1_hat, norm2_hat))
        force_closure = compute_force_closure(vertices_hat, normals_hat, num_facets, mu, gamma, object_mass)
        if force_closure:
            success_count += 1
    return success_count / NUM_TRIALS_ROBUST
    # the higher the better, ranges from [0, 1]


def compute_ferrari_canny(vertices, normals, num_facets, mu, gamma, object_mass):
    if not compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
        return 0 # not a force closure
    values = []
    rng = np.random.default_rng(RANDOM_SEED)
    # Monte Carlo Sampling
    for _ in range(NUM_TRIAlS_FER):
        wrench = normalize(rng.normal(size=(6,)))
        # print(f"wrench: {wrench}")
        try:
            problem, G, f = _opt_min_force(vertices, normals, num_facets, mu, gamma, wrench)
        except Exception as e:
            # raise Exception(f"Cannot compute for wrench = {wrench} even if the grasp is in force closure")
            return 0

        if problem.status == 'optimal' and np.isclose(np.linalg.norm(G @ f - wrench), 0, atol=ERROR_TOL):
            values.append(problem.value) # problem.value is the value for the l2 norm
        else:
            # raise Exception(f"Cannot compute for wrench = {wrench} even if the grasp is in force closure")
            return 0
    return 1/np.max(values) # no need for sqrt
    # the higher the better, ranges from (0, np.inf)


def get_gripper_pose(vertices, object_mesh): # you may or may not need this method 
    origin = np.mean(vertices, axis=0)
    direction = vertices[0] - vertices[1]

    up = np.array([0, 0, 1])
    y = normalize(direction)
    x = normalize(np.cross(up, y))
    z = np.cross(x, y)

    gripper_top = origin + GRIPPER_LENGTH * z
    gripper_double = origin + 2 * GRIPPER_LENGTH * z
    if len(find_intersections(object_mesh, gripper_top, gripper_double)[0]) > 0:
        z = normalize(np.cross(up, y))
        x = np.cross(y, x) # TODO: check this line
    result = np.eye(4)
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z
    result[0:3,3] = origin
    return result


def visualize_grasp(mesh, vertices, pose):
    p1, p2 = vertices
    center = (p1 + p2) / 2
    approach = pose[:3, 2]
    tail = center - GRIPPER_LENGTH * approach

    contact_points = []
    for v in vertices:
        contact_points.append(vedo.Point(pos=v, r=30))

    vec = (p1 - p2) / np.linalg.norm(p1 - p2)
    line = vedo.shapes.Tube([center + 0.5 * MAX_GRIPPER_DIST * vec,
                                   center - 0.5 * MAX_GRIPPER_DIST * vec], r=0.001, c='g')
    approach = vedo.shapes.Tube([center, tail], r=0.001, c='g')
    vedo.show([mesh, line, approach] + contact_points, new=True)

def randomly_sample_from_mesh(mesh, n):
    vertices, face_ind = sample_surface_even(mesh, n, None, RANDOM_SEED) # TODO: Evenly?
    normals = mesh.face_normals[face_ind]

    return vertices, normals

def sample_valid_vertices(mesh, n):
    table_height = -0.015 # TODO: change
    vertices = []
    seed = 0
    while len(vertices) < n:
        seed += 1
        vertice, _ = sample_surface_even(mesh, 1, None, RANDOM_SEED + seed)
        normal = normal_at_point(mesh, vertice[0])
        normal = normal[None, :]
        intersections, _ = find_intersections(mesh, vertice[0], vertice[0] - MAX_GRIPPER_DIST * normal[0])
        print("intersections:", len(intersections))
        print("distances:", np.linalg.norm(intersections - vertice, axis=-1))
        print("min_dist_req", MIN_GRIPPER_DIST)
        if len(intersections) == 0:
            continue
        mask = MIN_GRIPPER_DIST < np.linalg.norm(intersections - vertice, axis=-1)
        intersections = intersections[mask]
        if len(intersections) == 0:
            continue
        vertice_other = intersections[np.argmax(np.linalg.norm(intersections - vertice, axis=-1))]
        vertice_pair = np.vstack((vertice, vertice_other))
        assert vertice_pair.shape == (2, 3)
        vec = vertice_pair[0] - vertice_pair[1]
        if vec[2] > 0:
            vec = -vec
        dist = np.linalg.norm(vec)
        offset = (MAX_GRIPPER_DIST - dist) / 2
        if np.min(vertice_pair[:, 2]) + offset * vec[2] / dist < table_height:
            print(np.min(vertice_pair[:, 2]) + offset * vec[2], dist < table_height)
            continue
        if compute_force_closure(vertice_pair, [normal_at_point(mesh, vertice_pair[0]), normal_at_point(mesh, vertice_pair[1])], 64, 0.5, 0.1, 0.25):
            print(len(vertices))
            vertices.append(vertice_pair)
    return vertices

def load_mesh(object_name):
    mesh = trimesh.load_mesh('objects/{}.obj'.format(object_name))
    mesh.fix_normals()
    return mesh

if __name__ == '__main__':
    mesh = trimesh.load_mesh("object_mesh_dec11.stl")
    mesh.fix_normals()
    
    vertices = sample_valid_vertices(mesh, 100)
    print("sampled")
    metric_ferrari_canny = [compute_ferrari_canny(v, [normal_at_point(mesh, v[0]), normal_at_point(mesh, v[1])], 64, 0.5, 0.1, 0.25) for v in vertices]
    print("ferrari")
    metric_robust_fc = [compute_robust_force_closure(v, [normal_at_point(mesh, v[0]), normal_at_point(mesh, v[1])], 64, 0.5, 0.1, 0.25, mesh) for v in vertices]
    print("robust")
    metric_gravity = [compute_gravity_resistance(v, [normal_at_point(mesh, v[0]), normal_at_point(mesh, v[1])], 64, 0.5, 0.1, 0.25) for v in vertices]
    print("gravity")
    metric_ferrari_canny = np.array(metric_ferrari_canny)
    metric_robust_fc = np.array(metric_robust_fc)
    metric_gravity = np.array(metric_gravity)

    mask = (metric_ferrari_canny > 0) & (metric_gravity < np.inf)
    metric_ferrari_canny = metric_ferrari_canny[mask]
    metric_robust_fc = metric_robust_fc[mask]
    metric_gravity = metric_gravity[mask]
    vertices = np.array(vertices)[mask]
    
    standardized_ferrari_canny = (metric_ferrari_canny - np.mean(metric_ferrari_canny)) / np.std(metric_ferrari_canny)
    standardized_robust_fc = (metric_robust_fc - np.mean(metric_robust_fc)) / np.std(metric_robust_fc)
    standardized_gravity = -(metric_gravity - np.mean(metric_gravity)) / np.std(metric_gravity)
    std_score = standardized_ferrari_canny + standardized_robust_fc + standardized_gravity
    idx = np.argmax(std_score)
    pose = get_gripper_pose(vertices[idx], mesh)
    visualize_grasp(mesh, vertices[idx], pose)

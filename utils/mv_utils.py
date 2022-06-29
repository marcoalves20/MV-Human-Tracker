import torch
import numpy as np
from typing import List
from data_objects import Calib
import cv2
import time

def calc_pairwise_f_mats(calibs: List[Calib]):
    skew_op = lambda x: torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: torch.inverse(K_0).t() @ (
            R_0 @ R_1.t()) @ K_1.t() @ skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))

    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                    RT_1[:, :3], RT_1[:, 3])
    F = torch.zeros(len(calibs), len(calibs), 3, 3)  # NxNx3x3 matrix
    # TODO: optimize this stupid nested for loop
    for i in range(len(calibs)):
        for j in range(len(calibs)):
            F[i, j] += fundamental_RT_op(torch.tensor(calibs[i].K),
                                         torch.tensor(calibs[i].Rt),
                                         torch.tensor(calibs[j].K), torch.tensor(calibs[j].Rt))
            if F[i, j].sum() == 0:
                F[i, j] += 1e-12  # to avoid nan

    return F.numpy()

def projected_distance(pts_0, pts_1, F):
    """
    Compute point distance with epipolar geometry knowledge
    :param pts_0: numpy points array with shape Nx17x2
    :param pts_1: numpy points array with shape Nx17x2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """
    # lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2,
    #                                         F )  # I know 2 is not seems right, but it actually work for this dataset
    # lines = lines.reshape ( -1, 3 )
    # points_1 = np.ones ( (lines.shape[0], 3) )
    # points_1[:, :2] = pts_1.reshape((-1, 2))
    #
    # # to begin here!
    # dist = np.sum ( lines * points_1, axis=1 ) / np.linalg.norm ( lines[:, :2], axis=1 )
    # dist = np.abs ( dist )
    # dist = np.mean ( dist )

    lines = cv2.computeCorrespondEpilines(pts_0.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 1, 1, 3) # (-1, 17, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], 1, 3])
    points_1[0, :, 0, :2] = pts_1 # points_1[0, :, :, :2]

    dist = np.sum(lines * points_1, axis=3)  # / np.linalg.norm(lines[:, :, :, :2], axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)

    return dist

def geometry_affinity(points_set, Fs, dimGroup):
    M, _ = points_set.shape
    # distance_matrix = np.zeros ( (M, M), dtype=np.float32 )
    distance_matrix = np.ones((M, M), dtype=np.float32) * 50
    np.fill_diagonal(distance_matrix, 0)
    # TODO: remove this stupid nested for loop
    import time
    start_time = time.time()
    n_groups = len(dimGroup)
    for cam_id0, h in enumerate(range(n_groups - 1)):
        for cam_add, k in enumerate(range(cam_id0 + 1, n_groups - 1)):
            cam_id1 = cam_id0 + cam_add + 1
            # if there is no one in some view, skip it!
            if dimGroup[h] == dimGroup[h + 1] or dimGroup[k] == dimGroup[k + 1]:
                continue

            pose_id0 = points_set[dimGroup[h]:dimGroup[h + 1]]
            pose_id1 = points_set[dimGroup[k]:dimGroup[k + 1]]
            mean_dst = 0.5 * (projected_distance(pose_id0, pose_id1, Fs[cam_id0, cam_id1]) +
                              projected_distance(pose_id1, pose_id0, Fs[cam_id1, cam_id0]).T)
            distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]] = mean_dst
            # symmetric matrix
            distance_matrix[dimGroup[k]:dimGroup[k + 1], dimGroup[h]:dimGroup[h + 1]] = \
                distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]].T

    end_time = time.time()
    # print('using %fs' % (end_time - start_time))

    affinity_matrix = - (distance_matrix - distance_matrix.mean()) / distance_matrix.std()
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp(-5 * affinity_matrix))
    return distance_matrix, affinity_matrix


def transform_closure(x_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param x_bin: np.array which is binarized by a threshold
    :return:
    """
    temp = np.zeros_like(x_bin)
    N = x_bin.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                temp[i, j] = x_bin[i, j] or (x_bin[i, k] and x_bin[k, j])

    vis = np.zeros(N)
    match_result_mat = np.zeros_like(x_bin)
    for i, row in enumerate(temp):
        if vis[i]:
            continue
        for j, is_relative in enumerate(row):
            if is_relative:
                vis[j] = 1
                match_result_mat[j, i] = 1
    return match_result_mat


def match_als(W: np.ndarray, dimGroup, **kwargs):
    """
    % This function is to solve
    % min - <W,X> + alpha||X||_* + beta||X||_1, st. X \in C
    % The problem is rewritten as
    % <beta-W,AB^T> + alpha/2||A||^2 + alpha/2||B||^2
    % st AB^T=Z, Z\in\Omega
    % ---- Output:
    % X: a sparse binary matrix indicating correspondences
    % A: AA^T = X;
    % info: other info.
    % ---- Required input:
    % W: sparse input matrix storing scores of pairwise matches
    % dimGroup: a vector storing the number of points on each objects
    % ---- Other options:
    % maxRank: the restricted rank of X* (select it as large as possible)
    % alpha: the weight of nuclear norm
    % beta: the weight of l1 norm
    % pSelect: propotion of selected points, i.e., m'/m in section 5.4 in the paper
    % tol: tolerance of convergence
    % maxIter: maximal iteration
    % verbose: display info or not
    % eigenvalues: output eigenvalues or not
    """
    # optional paramters
    alpha = 50
    beta = 0.1
    # maxRank = max(dimGroup) * 4
    n_max_pp = np.diff(dimGroup)
    maxRank = max(n_max_pp) * 2
    # maxRank = max(n_max_pp) * 4

    pSelect = 1
    tol = 1e-4
    maxIter = 1000
    verbose = False
    eigenvalues = False
    W = 0.5 * (W + W.T)
    X = W.copy()
    Z = W.copy()
    Y = np.zeros_like(W)
    mu = 64
    n = X.shape[0]
    maxRank = min(n, maxRank)

    A = np.random.RandomState(0).rand(n, maxRank)

    iter_cnt = 0
    t0 = time.time()
    for iter_idx in range(maxIter):
        X0 = X.copy()
        X = Z - (Y - W + beta) / mu
        B = (np.linalg.inv(A.T @ A + alpha / mu * np.eye(maxRank)) @ (A.T @ X)).T
        A = (np.linalg.inv(B.T @ B + alpha / mu * np.eye(maxRank)) @ (B.T @ X.T)).T
        X = A @ B.T

        Z = X + Y / mu
        # enforce the self-matching to be null
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            Z[ind1:ind2, ind1:ind2] = 0

        if pSelect == 1:
            Z[np.arange(n), np.arange(n)] = 1

        Z[Z < 0] = 0
        Z[Z > 1] = 1

        Y = Y + mu * (X - Z)

        # test if convergence
        pRes = np.linalg.norm(X - Z) / n
        dRes = mu * np.linalg.norm(X - X0) / n
        if verbose:
            print('Iter = %d, Res = (%d,%d), mu = %d \n', iter, pRes, dRes, mu)

        if pRes < tol and dRes < tol:
            iter_cnt = iter_idx
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = 0.5 * (X + X.T)
    X_bin = X > 0.5

    total_time = time.time() - t0

    match_mat = transform_closure(X_bin)

    return match_mat, X_bin
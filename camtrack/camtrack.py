#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple, Union

import sys
import numpy as np
import scipy as sp
import cv2
from itertools import combinations
from collections import deque

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)


def triangulate_points_on_two_frames(frame_1: int, frame_2: int,
                                     corner_storage: CornerStorage,
                                     view_mats: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                                     intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    correspondences = build_correspondences(
        corner_storage[frame_1],
        corner_storage[frame_2]
    )

    view_mat_1, view_mat_2 = None, None
    if (isinstance(view_mats, list)):
        view_mat_1 = view_mats[frame_1]
        view_mat_2 = view_mats[frame_2]
    else:
        view_mat_1, view_mat_2 = view_mats

    points3d, ids, _ = triangulate_correspondences(
        correspondences,
        view_mat_1,
        view_mat_2,
        intrinsic_mat,
        TriangulationParameters(
            max_reprojection_error = 5.0,
            min_triangulation_angle_deg = 1.0,
            min_depth = 0.01
        )
    )

    return ids, points3d


def triangulate_points_on_multiple_frames(frame_ids: np.ndarray,
                                          corner_storage: CornerStorage,
                                          view_mats: List[np.ndarray],
                                          intrinsic_mat: np.ndarray,
                                          frames_cnt_threshold: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    K = np.zeros((4, 4))
    K[: 3, : 3] = intrinsic_mat
    K[2][2] = 0
    K[2][3] = 1
    K[3][2] = 1

    good_point_ids = set()
    for i, j in combinations(frame_ids, 2):
        cur_ids, _ = triangulate_points_on_two_frames(i, j, corner_storage, view_mats, intrinsic_mat)
        for ind in cur_ids:
            good_point_ids.add(ind)

    def calc_full_mat(frame: int, view_mats: List[np.ndarray], K: np.ndarray):
        cam_mat = np.vstack((view_mats[frame].astype(float), np.array([0, 0, 0, 1], dtype = float)))
        return K @ cam_mat


    full_mats = dict()
    ids, points3d = [], []

    for pt in good_point_ids:
        frames_to_triangulate = []
        for frame in frame_ids:
            if (not (pt in corner_storage[frame].ids.tolist())):
                continue

            ind_of_point_in_corner_storage = corner_storage[frame].ids.tolist().index(pt)
            frames_to_triangulate.append((frame, ind_of_point_in_corner_storage))

        if (len(frames_to_triangulate) < frames_cnt_threshold):
            continue


        least_sq_system = []
        for frame, ind_of_point_in_corner_storage in frames_to_triangulate:
            fm = full_mats.get(frame)
            if (fm is None):
                full_mats[frame] = calc_full_mat(frame, view_mats, K)
                fm = full_mats[frame]

            least_sq_system.append(fm[3] * corner_storage[frame].points[ind_of_point_in_corner_storage][0] - fm[0])
            least_sq_system.append(fm[3] * corner_storage[frame].points[ind_of_point_in_corner_storage][1] - fm[1])

        least_sq_system = np.array(least_sq_system)
        coordinates = sp.linalg.lstsq(
            least_sq_system[:, : 3],
            -least_sq_system[:, 3],
            lapack_driver = "gelsy",
            check_finite = False
        )[0]

        ids.append(pt)
        points3d.append(coordinates)

    return np.array(ids).astype(np.int64), np.array(points3d)


def get_new_points_for_frame_with_two_frames(frame: int, frame_1: int, frame_2: int,
                                             corner_storage: CornerStorage,
                                             view_mats: List[np.ndarray],
                                             intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    ids_1, points3d_1 = triangulate_points_on_two_frames(
        frame, frame_1,
        corner_storage,
        view_mats,
        intrinsic_mat
    )

    ids_2, points3d_2 = triangulate_points_on_two_frames(
        frame, frame_2,
        corner_storage,
        view_mats,
        intrinsic_mat
    )

    if (len(ids_1) < len(ids_2)):
        ids_1, ids_2 = ids_2, ids_1
        points3d_1, points3d_2 = points3d_2, points3d_1

    return ids_1, points3d_1


def random_update_point_cloud(frame_count: int,
                              corner_storage: CornerStorage,
                              view_mats: List[np.ndarray],
                              intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    RANDOM_FRAMES_TO_RETRIANGULATE = 5
    assert(frame_count >= RANDOM_FRAMES_TO_RETRIANGULATE)

    frame_ids = np.random.choice(np.arange(0, frame_count), RANDOM_FRAMES_TO_RETRIANGULATE, replace = False)

    return triangulate_points_on_multiple_frames(frame_ids, corner_storage, view_mats, intrinsic_mat)


def get_best_frame_to_process(frames_to_process: List[int],
                              corner_storage: CornerStorage,
                              point_cloud_builder: PointCloudBuilder,
                              ransac_parameters: dict) -> int:

    def get(frames_to_process):
        best = (-1, None)
        
        for frame in frames_to_process:
            _, point_builder_ind, corners_ind = np.intersect1d(
                point_cloud_builder.ids,
                corner_storage[frame].ids,
                return_indices = True
            )

            real_points = point_cloud_builder.points[point_builder_ind]
            proj_points = corner_storage[frame].points[corners_ind]
            assert(len(real_points) == len(proj_points))

            if (len(real_points) < 4):
                continue

            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints = real_points,
                imagePoints = proj_points,
                **ransac_parameters
            )

            if (not retval):
                continue

            if (inliers.size > best[0]):
                best = (inliers.size, frame)

        return best[1]


    res = get(np.random.choice(frames_to_process, max(min(len(frames_to_process), 50), len(frames_to_process) // 10), replace = False))

    if (res is None):
        res = get(frames_to_process)

    return res


def calc_camera_positions(iteration: int,
                          frame_1: int, frame_2: int, frame_count: int,
                          corner_storage: CornerStorage,
                          rotations: List[np.ndarray], translations: List[np.ndarray],
                          view_mats: List[np.ndarray],
                          intrinsic_mat: np.ndarray,
                          point_cloud_builder: PointCloudBuilder,
                          ransac_parameters: dict,
                          ransac_parameters_for_shrinking: Optional[dict] = None,
                          ransac_parameters_for_refining: Optional[dict] = None) -> None:

    print(f"\n\n\nIteration #{iteration}")


    frames_to_process = [
        *range(frame_1 + 1, frame_2),
        *range(frame_1 - 1, -1, -1),
        *range(frame_2 + 1, frame_count)
    ]
    bad_frames = []

    frames_processed = [frame_1, frame_2]
    while (len(frames_to_process) > 0):
        frame = get_best_frame_to_process(
            frames_to_process,
            corner_storage,
            point_cloud_builder,
            ransac_parameters
        )
        assert(not (frame is None))
        print(f"Processing frame #{frame}")
        frames_to_process.remove(frame)

        _, point_builder_ind, corners_ind = np.intersect1d(
            point_cloud_builder.ids,
            corner_storage[frame].ids,
            return_indices = True
        )

        real_points = point_cloud_builder.points[point_builder_ind]
        proj_points = corner_storage[frame].points[corners_ind]
        assert(len(real_points) == len(proj_points))

        if (len(real_points) < 4):
            print(f"Failed to process frame #{frame}")
            bad_frames.append(frame)
            continue

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = real_points,
            imagePoints = proj_points,
            **ransac_parameters
        )

        if (not retval):
            print(f"Failed to process frame #{frame}")
            bad_frames.append(frame)
            continue

        print(f"Found {len(inliers)} inliers")

        _, rvec, tvec = cv2.solvePnP(
            objectPoints = real_points[inliers],
            imagePoints = proj_points[inliers],
            cameraMatrix = intrinsic_mat,
            distCoeffs = None
        )

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        point_cloud_builder.remove_points(outliers)

        rotations[frame] = rvec
        translations[frame] = tvec
        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        ids, points3d = get_new_points_for_frame_with_two_frames(
            frame, frame_1, frame_2,
            corner_storage,
            view_mats,
            intrinsic_mat
        )
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        for shift in [15, 30, 45]:
            if (len(frames_processed) >= shift * 2):
                ids, points3d = get_new_points_for_frame_with_two_frames(
                    frame, frames_processed[-shift], frames_processed[-shift * 2],
                    corner_storage,
                    view_mats,
                    intrinsic_mat
                )
                point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

            if ((frame - shift) in frames_processed and (frame - shift * 2) in frames_processed):
                ids, points3d = get_new_points_for_frame_with_two_frames(
                    frame, frame - shift, frame - shift * 2,
                    corner_storage,
                    view_mats,
                    intrinsic_mat
                )
                point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}\n")

        frames_processed.append(frame)

        if (len(frames_to_process) == 0):
            frames_to_process = bad_frames
            bad_frames = []


    print("\n\nExtending point cloud")

    EXTENDING_POINT_CLOUD_ITERATIONS = 5
    for it in range(EXTENDING_POINT_CLOUD_ITERATIONS):
        print("Attempt #", it + 1, sep = "")

        ids, points3d = random_update_point_cloud(frame_count, corner_storage, view_mats, intrinsic_mat)
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}\n")


    print(f"\n\nRetriangulating points")

    retriangulate_frames = np.arange(0, frame_count, int(np.round(frame_count / 5)))
    #retriangulate_frames = np.concatenate((retriangulate_frames, np.array([frame_1, frame_2])))
    #retriangulate_frames = np.unique(retriangulate_frames)

    ids, points3d = triangulate_points_on_multiple_frames(retriangulate_frames, corner_storage, view_mats, intrinsic_mat)
    point_cloud_builder.add_points(ids, points3d, recalc_positions = True)

    print(f"Retriangulated {len(ids)} points")


    if (not (ransac_parameters_for_refining is None)):
        print("\n\nRefining positions")

        for frame in range(frame_count):
            print(f"Processing frame #{frame}")

            _, point_builder_ind, corners_ind = np.intersect1d(
                point_cloud_builder.ids,
                corner_storage[frame].ids,
                return_indices = True
            )

            real_points = point_cloud_builder.points[point_builder_ind]
            proj_points = corner_storage[frame].points[corners_ind]
            assert(len(real_points) == len(proj_points))
            if (len(real_points) < 4):
                continue

            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints = real_points,
                imagePoints = proj_points,
                **ransac_parameters_for_refining
            )

            if (not retval):
                print(f"Failed to process frame #{frame}")
                continue

            _, rvec, tvec = cv2.solvePnP(
                objectPoints = real_points[inliers],
                imagePoints = proj_points[inliers],
                cameraMatrix = intrinsic_mat,
                distCoeffs = None
            )

            rotations[frame] = rvec
            translations[frame] = tvec
            view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)


    if (not (ransac_parameters_for_shrinking is None)):
        print("\n\nShrinking outliers")

        for frame in range(frame_count):
            print(f"Processing frame #{frame}")

            _, point_builder_ind, corners_ind = np.intersect1d(
                point_cloud_builder.ids,
                corner_storage[frame].ids,
                return_indices = True
            )

            real_points = point_cloud_builder.points[point_builder_ind]
            proj_points = corner_storage[frame].points[corners_ind]
            assert(len(real_points) == len(proj_points))
            if (len(real_points) < 4):
                continue

            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints = real_points,
                imagePoints = proj_points,
                **ransac_parameters_for_shrinking
            )

            if (not retval):
                print(f"Failed to process frame #{frame}")
                continue

            outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
            point_cloud_builder.remove_points(outliers)

            print(f"Point cloud size is {len(point_cloud_builder.ids)}\n")


def find_good_frame_pair(frame_count: int,
                         corner_storage: CornerStorage,
                         intrinsic_mat: np.ndarray,
                         reproj_threshold: float) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:

    print("Finding good frame pair")

    step = 10
    if (frame_count <= 60):
        step = 1

    def check_frame_pair_is_good_enough(frame_1: int,
                                        frame_2: int,
                                        corner_storage: CornerStorage,
                                        intrinsic_mat: np.ndarray,
                                        reproj_threshold: float):

        correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
        if (correspondences.ids.size < 10):
            return None

        H, mask_homography = cv2.findHomography(
            correspondences.points_1,
            correspondences.points_2,
            method = cv2.RANSAC,
            maxIters = 5000,
            ransacReprojThreshold = reproj_threshold,
            confidence = 0.999999
        )

        E, mask_essential_mat = cv2.findEssentialMat(
            correspondences.points_1,
            correspondences.points_2,
            intrinsic_mat,
            method = cv2.RANSAC,
            maxIters = 5000,
            threshold = reproj_threshold,
            prob = 0.999999
        )

        if (mask_essential_mat.astype(int).sum() <= mask_homography.astype(int).sum() * 2):
            return None

        _, R, t, _ = cv2.recoverPose(
            E,
            correspondences.points_1,
            correspondences.points_2,
            intrinsic_mat,
        )

        view_mat_1 = pose_to_view_mat3x4(Pose(r_mat = np.eye(3), t_vec = np.zeros(3)))
        view_mat_2 = pose_to_view_mat3x4(Pose(r_mat = R.T, t_vec = -R.T @ t))

        _, ids, median_cos = triangulate_correspondences(
            correspondences,
            view_mat_1,
            view_mat_2,
            intrinsic_mat,
            TriangulationParameters(
                max_reprojection_error = 5.0,
                min_triangulation_angle_deg = 1.0,
                min_depth = 0.01
            )
        )

        if (median_cos > np.cos(3 / 180 * np.pi)):
            return None

        if (ids.size < 100):
            return None

        return ids.size, median_cos

    count, median_cos = [], []
    frame_pairs = []
    for frame_1 in range(frame_count):
        print(f"Processing frame {frame_1}")

        for frame_2 in range(min(frame_count, frame_1 + step), frame_count, step):
            res = check_frame_pair_is_good_enough(frame_1, frame_2, corner_storage, intrinsic_mat, reproj_threshold)

            if (res is None):
                continue

            count.append(res[0])
            median_cos.append(res[1])
            frame_pairs.append((frame_1, frame_2, res[0], res[1]))

            #cnt, median_cos = res
            
            #if (best[0] is None or best[0] < cnt):
            #if (best is None or best[0] > median_cos):
            #    best = (median_cos, cnt, frame_1, frame_2)

    count = np.array(sorted(count))
    median_cos = np.array(sorted(median_cos))

    best = None
    for frame_1, frame_2, cnt, cos in frame_pairs:
        rank = (np.searchsorted(count, cnt) + 1) * (median_cos.size - np.searchsorted(median_cos, cos) + 1)

        if (best is None or rank > best[0]):
            best = (rank, cnt, cos, frame_1, frame_2)

    if (best is None):
        raise Exception("Error! Can't find good pair of frames")

    _, _, _, frame_1, frame_2 = best
    
    print(f"Found pair ({frame_1}, {frame_2}) with {best[1]} inliers and median angle {np.arccos(best[2]) / np.pi * 180} degrees")

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])

    E, mask = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        method = cv2.RANSAC,
        maxIters = 10000,
        threshold = reproj_threshold,
        prob = 0.999999
    )

    '''
    for i in range(0, 25):
        E, mask = cv2.findEssentialMat(
            correspondences.points_1,
            correspondences.points_2,
            intrinsic_mat,
            method = cv2.RANSAC,
            maxIters = 10000,
            threshold = i,
            prob = 0.999999
        )
        print(i, mask.sum())

    assert(0)
    '''

    _, R, t, _ = cv2.recoverPose(
        E,
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
    )

    return (frame_1, Pose(r_mat = np.eye(3), t_vec = np.zeros(3))), (frame_2, Pose(r_mat = R.T, t_vec = -R.T @ t))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(corner_storage)
    assert(frame_count == len(rgb_sequence))
    print(f"Processing {frame_count} frames")

    thr = None

    if ("ironman_translation_fast" in frame_sequence_path):
        thr = 7 # 12
    
    if ("room" in frame_sequence_path):
        thr = 1

    if ("bike_translation_slow" in frame_sequence_path):
        thr = 1 # 12

    if ("house_free_motion" in frame_sequence_path):
        thr = 1 # 12

    if ("soda_free_motion" in frame_sequence_path):
        thr = 3 # 5

    if ("fox_head_short" in frame_sequence_path):
        thr = 1

    if ("fox_head_full" in frame_sequence_path):
        thr = 1

    #print(frame_sequence_path, file = sys.stderr)
    thr = float(thr)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_good_frame_pair(frame_count, corner_storage, intrinsic_mat, thr)

    print(f"Using frames {known_view_1[0]} and {known_view_2[0]}")

    np.random.seed(4824)

    if (known_view_1[0] > known_view_2[0]):
        known_view_1, known_view_2 = known_view_2, known_view_1

    frame_1 = known_view_1[0]
    frame_2 = known_view_2[0]

    rotations = [None] * frame_count
    translations = [None] * frame_count
    view_mats = [None] * frame_count
    point_cloud_builder = PointCloudBuilder()

    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    rotations[frame_1] = cv2.Rodrigues(view_mat_1[:, : 3])[0]
    translations[frame_1] = np.array(view_mat_1[:, 3 :])
    view_mats[frame_1] = view_mat_1

    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    rotations[frame_2] = cv2.Rodrigues(view_mat_2[:, : 3])[0]
    translations[frame_2] = np.array(view_mat_2[:, 3 :])
    view_mats[frame_2] = view_mat_2

    print("Building init point cloud\n")

    ids, points3d = triangulate_points_on_two_frames(
        frame_1, frame_2,
        corner_storage,
        view_mats,
        intrinsic_mat
    )
    point_cloud_builder.add_points(ids, points3d)

    def RANSAC_PARAMETERS(reprojection_error, confidence_level = 0.999999):
        return dict(
            cameraMatrix = intrinsic_mat,
            reprojectionError = reprojection_error,
            confidence = confidence_level,
            iterationsCount = 1000,
            distCoeffs = None
        )

    reproj_errors = [5, 12, 20, 24]
    ransac_parameters_for_refining = [None, RANSAC_PARAMETERS(5)]
    ransac_parameters_for_shrinking = [None, RANSAC_PARAMETERS(24, 0.5)]

    for iteration in range(len(reproj_errors)):
        calc_camera_positions(
            iteration + 1,
            frame_1, frame_2, frame_count,
            corner_storage,
            rotations, translations,
            view_mats,
            intrinsic_mat,
            point_cloud_builder,
            RANSAC_PARAMETERS(reprojection_error = reproj_errors[iteration]),
            ransac_parameters_for_shrinking = ransac_parameters_for_shrinking[int(iteration == len(reproj_errors) - 1)],
            ransac_parameters_for_refining = ransac_parameters_for_refining[int(iteration == len(reproj_errors) - 1)]
        )

    print("\nCamera positions calculated")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

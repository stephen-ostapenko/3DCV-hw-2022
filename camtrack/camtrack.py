#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

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
                                     view_mats: List[np.ndarray],
                                     intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    correspondences = build_correspondences(
        corner_storage[frame_1],
        corner_storage[frame_2]
    )

    points3d, ids, _ = triangulate_correspondences(
        correspondences,
        view_mats[frame_1],
        view_mats[frame_2],
        intrinsic_mat,
        TriangulationParameters(
            max_reprojection_error = 5.0,
            min_triangulation_angle_deg = 1.0,
            min_depth = 0.01
        )
    )

    return ids, points3d


def calc_full_mat(frame: int, view_mats: List[np.ndarray], K: np.ndarray):
    cam_mat = np.vstack((view_mats[frame].astype(float), np.array([0, 0, 0, 1], dtype = float)))
    return K @ cam_mat


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
        coordinates = sp.linalg.lstsq(least_sq_system[:, : 3], -least_sq_system[:, 3], lapack_driver = "gelsy", check_finite = False)[0]

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


    frames_to_process = deque([
        *range(frame_1 + 1, frame_2),
        *range(frame_1 - 1, -1, -1),
        *range(frame_2 + 1, frame_count)
    ])

    frames_processed = []
    while (len(frames_to_process) > 0):
        frame = frames_to_process.popleft()
        print(f"Processing frame #{frame}")

        _, point_builder_ind, corners_ind = np.intersect1d(
            point_cloud_builder.ids,
            corner_storage[frame].ids,
            return_indices = True
        )

        real_points = point_cloud_builder.points[point_builder_ind]
        proj_points = corner_storage[frame].points[corners_ind]
        assert(len(real_points) == len(proj_points))

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = real_points,
            imagePoints = proj_points,
            **ransac_parameters
        )

        if (not retval):
            print(f"Failed to process frame #{frame}")
            frames_to_process.append(frame)
            continue

        print(f"Found {len(inliers)} inliers")

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

        if (len(frames_processed) >= 30):
            ids, points3d = get_new_points_for_frame_with_two_frames(
                frame, frames_processed[-15], frames_processed[-30],
                corner_storage,
                view_mats,
                intrinsic_mat
            )
            point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        if (len(frames_processed) >= 90):
            ids, points3d = get_new_points_for_frame_with_two_frames(
                frame, frames_processed[-60], frames_processed[-90],
                corner_storage,
                view_mats,
                intrinsic_mat
            )
            point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}\n")

        frames_processed.append(frame)


    print("\n\nExtending point cloud")

    for it in range(5):
        print("Attempt #", it + 1)

        ids, points3d = random_update_point_cloud(frame_count, corner_storage, view_mats, intrinsic_mat)
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}\n")


    print(f"\n\nRetriangulating points")

    retriangulate_frames = np.arange(0, frame_count, int(np.round(frame_count / 5)))
    retriangulate_frames = np.concatenate((retriangulate_frames, np.array([frame_1, frame_2])))
    retriangulate_frames = np.unique(retriangulate_frames)

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

            '''
            real_points = real_points[inliers]
            proj_points = proj_points[inliers]

            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints = real_points,
                imagePoints = proj_points,
                cameraMatrix = intrinsic_mat,
                distCoeffs = None,
                rvec = rvec,
                tvec = tvec
            )
            '''

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


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:

    if (known_view_1 is None or known_view_2 is None):
        raise NotImplementedError()

    np.random.seed(824)

    if (known_view_1[0] > known_view_2[0]):
        known_view_1, known_view_2 = known_view_2, known_view_1

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
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

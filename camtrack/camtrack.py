#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

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


def triangulate_points_on_two_frames(frame1: int, frame2: int, \
                                     corner_storage: CornerStorage, \
                                     view_mats: List[np.ndarray], \
                                     intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points3d, ids, _ = triangulate_correspondences(
        correspondences,
        view_mats[frame1],
        view_mats[frame2],
        intrinsic_mat,
        TriangulationParameters(
            max_reprojection_error = 5.0,
            min_triangulation_angle_deg = 1.0,
            min_depth = 0.01
        )
    )

    return points3d, ids


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    if (known_view_1[0] > known_view_2[0]):
        known_view_1, known_view_2 = known_view_2, known_view_1

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()
    points3d, ids = triangulate_points_on_two_frames(known_view_1[0], known_view_2[0], \
                                                     corner_storage, \
                                                     view_mats, \
                                                     intrinsic_mat)
    point_cloud_builder.add_points(ids, points3d)

    RANSAC_PARAMETERS = dict(
        cameraMatrix = intrinsic_mat,
        reprojectionError = 5.0,
        confidence = 1 - 1e-6,
        iterationsCount = 1000,
        distCoeffs = None
    )

    for frame in range(known_view_1[0] + 1, known_view_2[0]):
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
            **RANSAC_PARAMETERS
        )

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        point_cloud_builder.remove_points(outliers)

        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        points3d_1, ids_1 = triangulate_points_on_two_frames(known_view_1[0], frame, \
                                                             corner_storage, \
                                                             view_mats, \
                                                             intrinsic_mat)
        points3d_2, ids_2 = triangulate_points_on_two_frames(frame, known_view_2[0], \
                                                             corner_storage, \
                                                             view_mats, \
                                                             intrinsic_mat)
        if (len(ids_1) < len(ids_2)):
            ids_1, ids_2 = ids_2, ids_1
            points3d_1, points3d_2 = points3d_2, points3d_1

        point_cloud_builder.add_points(ids_1, points3d_1, recalc_positions = False)

    for frame in range(frame_count):
        if (not (view_mats[frame] is None)):
            continue

        print("kek", frame)

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
            **RANSAC_PARAMETERS
        )

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        print(outliers)
        point_cloud_builder.remove_points(outliers)

        '''_, point_builder_ind, corners_ind = np.intersect1d(
            point_cloud_builder.ids,
            corner_storage[frame].ids,
            return_indices = True
        )

        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints = real_points,
            imagePoints = proj_points,
            cameraMatrix = intrinsic_mat,
            distCoeffs = None,
            rvec = rvec,
            tvec = tvec
        )'''

        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        points3d, ids = triangulate_points_on_two_frames(known_view_2[0], frame, corner_storage, view_mats, intrinsic_mat)
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

    print("my job here is done")

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

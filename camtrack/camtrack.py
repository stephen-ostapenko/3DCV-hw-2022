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


def triangulate_points_on_two_frames(frame_1: int, frame_2: int, \
                                     corner_storage: CornerStorage, \
                                     view_mats: List[np.ndarray], \
                                     intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
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


def get_best_points_for_frame(frame: int, frame_1: int, frame_2: int ,\
                              corner_storage: CornerStorage, \
                              view_mats: List[np.ndarray], \
                              intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    ids_1, points3d_1 = triangulate_points_on_two_frames(frame, frame_1, \
                                                         corner_storage, \
                                                         view_mats, \
                                                         intrinsic_mat)
    ids_2, points3d_2 = triangulate_points_on_two_frames(frame, frame_2, \
                                                         corner_storage, \
                                                         view_mats, \
                                                         intrinsic_mat)
    if (len(ids_1) < len(ids_2)):
        ids_1, ids_2 = ids_2, ids_1
        points3d_1, points3d_2 = points3d_2, points3d_1

    return ids_1, points3d_1


def calc_camera_positions(iteration: int, \
                          frame_1: int, frame_2: int, frame_count: int, \
                          corner_storage: CornerStorage, \
                          rotations: List[np.ndarray], translations: List[np.ndarray], \
                          view_mats: List[np.ndarray], \
                          intrinsic_mat: np.ndarray, \
                          point_cloud_builder: PointCloudBuilder, \
                          ransac_parameters: dict, \
                          shrink_outliers: bool) -> None:

    for frame in range(frame_1 + 1, frame_2):
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

        print(f"Found {len(inliers)} inliers")

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        point_cloud_builder.remove_points(outliers)

        rotations[frame] = rvec
        translations[frame] = tvec
        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        ids, points3d = get_best_points_for_frame(frame, frame_1, frame_2, corner_storage, view_mats, intrinsic_mat)
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}")


    for frame in [*range(frame_1 - 1, -1, -1), *range(frame_2 + 1, frame_count)]:
        if (not (view_mats[frame] is None)):
            continue

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

        print(f"Found {len(inliers)} inliers")

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        point_cloud_builder.remove_points(outliers)

        rotations[frame] = rvec
        translations[frame] = tvec
        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        ids, points3d = get_best_points_for_frame(frame, frame_1, frame_2, corner_storage, view_mats, intrinsic_mat)
        point_cloud_builder.add_points(ids, points3d, recalc_positions = False)

        print(f"Triangulated {len(ids)} points")
        print(f"Point cloud size is {len(point_cloud_builder.ids)}")


    if (not shrink_outliers):
        return

    print("\nShrinking outliers")

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

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = real_points,
            imagePoints = proj_points,
            **ransac_parameters
        )

        outliers = np.setdiff1d(np.arange(0, len(real_points)), inliers.T, assume_unique = True)
        point_cloud_builder.remove_points(outliers)


    print("\nRefining positions")

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

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = real_points,
            imagePoints = proj_points,
            **ransac_parameters
        )

        rotations[frame] = rvec
        translations[frame] = tvec
        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:

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

    ids, points3d = triangulate_points_on_two_frames(frame_1, frame_2, \
                                                     corner_storage, \
                                                     view_mats, \
                                                     intrinsic_mat
    )
    point_cloud_builder.add_points(ids, points3d)

    def RANSAC_PARAMETERS(reprojection_error):
        return dict(
            cameraMatrix = intrinsic_mat,
            reprojectionError = reprojection_error,
            confidence = 1 - 1e-6,
            iterationsCount = 1000,
            distCoeffs = None
        )

    for iteration in range(1, 4):
        calc_camera_positions(iteration, \
                              frame_1, frame_2, frame_count, \
                              corner_storage, \
                              rotations, translations, \
                              view_mats, \
                              intrinsic_mat, \
                              point_cloud_builder, \
                              RANSAC_PARAMETERS(reprojection_error = ([0, 5, 10, 48])[iteration]), \
                              shrink_outliers = (iteration == 3)
        )

    print("Camera positions calculated")

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

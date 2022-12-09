#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    filter_frame_corners,
    build_mask_for_corners,
    _to_int_tuple,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    MAX_CORNERS_CNT = 5000
    QUALITY_LEVEL = 0.05
    MIN_DISTANCE = 7
    FEATURE_SIZE = 8
    BLOCK_SIZE = 7

    LK_WINDOW_SIZE = (15, 15)
    LK_MAX_LEVEL = 4
    LK_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01)
    LK_MIN_EIG_THRESHOLD = 5 * 10 ** (-4)

    last_image = frame_sequence[0]
    features_pos = cv2.goodFeaturesToTrack(
        image = last_image,
        maxCorners = MAX_CORNERS_CNT,
        qualityLevel = QUALITY_LEVEL,
        minDistance = MIN_DISTANCE,
        blockSize = BLOCK_SIZE
    )
    corners = FrameCorners(
        ids = np.arange(0, len(features_pos)),
        points = features_pos,
        sizes = np.array([FEATURE_SIZE] * len(features_pos))
    )
    builder.set_corners_at_frame(0, corners)

    for index, next_image in enumerate(frame_sequence[1:], 1):
        features_pos, mask, error = cv2.calcOpticalFlowPyrLK(
            prevImg = np.uint8(last_image * 255),
            nextImg = np.uint8(next_image * 255),
            prevPts = corners.points,
            nextPts = None,
            winSize = LK_WINDOW_SIZE,
            maxLevel = LK_MAX_LEVEL,
            criteria = LK_CRITERIA,
            minEigThreshold = LK_MIN_EIG_THRESHOLD
        )
        corners.update_points_pos(features_pos)

        if (mask is None):
            mask = np.array([])
        else:
            mask = (mask == 1).flatten()

        if (mask.size and corners.cnt):
            corners = filter_frame_corners(corners, mask)

        if (MAX_CORNERS_CNT > corners.cnt):
            mask = build_mask_for_corners(next_image.shape, corners, MIN_DISTANCE)

            new_features = cv2.goodFeaturesToTrack(
                image = next_image,
                maxCorners = MAX_CORNERS_CNT - corners.cnt,
                qualityLevel = QUALITY_LEVEL,
                minDistance = MIN_DISTANCE,
                blockSize = BLOCK_SIZE,
                mask = mask
            )

            if (not (new_features is None)):
                corners.add_points(new_features, FEATURE_SIZE)

        builder.set_corners_at_frame(index, corners)
        last_image = next_image


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

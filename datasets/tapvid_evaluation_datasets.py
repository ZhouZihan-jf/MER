# Unmodified combination of:
#   - https://github.com/deepmind/tapnet/blob/b8e0641e3c6a3483060e49df1def87fef16c8d1a/evaluation_datasets.py
#   - https://github.com/deepmind/tapnet/blob/b8e0641e3c6a3483060e49df1def87fef16c8d1a/utils/transforms.py
#
#
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation dataset creation functions."""

from typing import Iterable, Mapping, Tuple, Union, Sequence

import mediapy as media
import numpy as np


DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

TRAIN_SIZE = (24, 256, 256, 3)

# from tapnet.utils import transforms
def convert_grid_coordinates(
    coords: np.ndarray,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> np.ndarray:
  """Convert image coordinates between image grids of different sizes.

  By default, it assumes that the image corners are aligned.  Therefore,
  it adds .5 (since (0,0) is assumed to be the center of the upper-left grid
  cell), multiplies by the size ratio, and then subtracts .5.

  Args:
    coords: The coordinates to be converted.  It is of shape [..., 2] if
      coordinate_format is 'xy' or [..., 3] if coordinate_format is 'tyx'.
    input_grid_size: The size of the image/grid that the coordinates currently
      are with respect to.  This is a 2-tuple of the format [width, height]
      if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    output_grid_size: The size of the target image/grid that you want the
      coordinates to be with respect to.  This is a 2-tuple of the format
      [width, height] if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    coordinate_format: Which format the coordinates are in.  This can be one
      of 'xy' (the default) or 'tyx', which are the only formats used in this
      project.

  Returns:
    The transformed coordinates, of the same shape as coordinates.

  Raises:
    ValueError: if coordinates don't match the given format.
  """
  if isinstance(input_grid_size, tuple):
    input_grid_size = np.array(input_grid_size)
  if isinstance(output_grid_size, tuple):
    output_grid_size = np.array(output_grid_size)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.')
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.')
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, TRAIN_SIZE[1:3])


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    additional_pck_thresholds: Iterable[float] = (),
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number
       of frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
       in the format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same
       format as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the
       same format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
      additional_pck_thresholds: additional thresholds that will result in
        pts_within_{x} for x in additional_pck_thresholds

  """

  metrics = {}

  # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
  # replicate it by indexing into an identity matrix.
  one_hot_eye = np.eye(gt_tracks.shape[2])
  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = one_hot_eye[query_frame] == 0

  # If we're using the first point on the track as a query, don't evaluate the
  # other points.
  if query_mode == 'first':
    for i in range(gt_occluded.shape[0]):
      index = np.where(gt_occluded[i] == 0)[0][0]
      evaluation_points[i, :index] = False
  elif query_mode != 'strided':
    raise ValueError('Unknown query mode ' + query_mode)

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=(1, 2),
  ) / np.sum(evaluation_points)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=(1, 2),
    )
    count_visible_points = np.sum(
        visible & evaluation_points, axis=(1, 2))
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=(1, 2))

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  for thresh in additional_pck_thresholds:
    within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)
    count_correct = np.sum(is_correct & evaluation_points, axis=(1, 2))
    count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct

  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics


def latex_table(mean_scalars: Mapping[str, float]) -> str:
  """Generate a latex table for displaying TAP-Vid and PCK metrics."""
  if 'average_jaccard' in mean_scalars:
    latex_fields = [
        'average_jaccard',
        'average_pts_within_thresh',
        'occlusion_accuracy',
        'jaccard_1',
        'jaccard_2',
        'jaccard_4',
        'jaccard_8',
        'jaccard_16',
        'pts_within_1',
        'pts_within_2',
        'pts_within_4',
        'pts_within_8',
        'pts_within_16',
    ]
    header = ('AJ & $<\\delta^{x}_{avg}$ & OA & Jac. $\\delta^{0}$ & ' +
              'Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & ' +
              'Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & $<\\delta^{0}$ & ' +
              '$<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & ' +
              '$<\\delta^{4}$')
  else:
    latex_fields = ['PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5']
    header = ' & '.join(latex_fields)

  body = ' & '.join(
      [f'{float(np.array(mean_scalars[x]*100)):.3}' for x in latex_fields])
  return '\n'.join([header, body])


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, sample queries
  strided every query_stride frames, ignoring points that are not visible
  at the selected frames.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.
    query_stride: When sampling query points, search for un-occluded points
      every query_stride frames and convert each one into a query.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3].  The video
        has floats scaled to the range [-1, 1].
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1].
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1].
      trackgroup: Index of the original track that each query point was
        sampled from.  This is useful for visualization.
  """
  tracks = []
  occs = []
  queries = []
  trackgroups = []
  total = 0
  trackgroup = np.arange(target_occluded.shape[0])
  for i in range(0, target_occluded.shape[1], query_stride):
    mask = target_occluded[:, i] == 0
    query = np.stack(
        [
            i * np.ones(target_occluded.shape[0:1]), target_points[:, i, 1],
            target_points[:, i, 0]
        ],
        axis=-1,
    )
    queries.append(query[mask])
    tracks.append(target_points[mask])
    occs.append(target_occluded[mask])
    trackgroups.append(trackgroup[mask])
    total += np.array(np.sum(target_occluded[:, i] == 0))

  return {
      'video':
          frames[np.newaxis, ...],
      'query_points':
          np.concatenate(queries, axis=0)[np.newaxis, ...],
      'target_points':
          np.concatenate(tracks, axis=0)[np.newaxis, ...],
      'occluded':
          np.concatenate(occs, axis=0)[np.newaxis, ...],
      'trackgroup':
          np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
  }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, use the first
  visible point in each track as the query.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3]
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1]
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1]
  """

  valid = np.sum(~target_occluded, axis=1) > 0
  target_points = target_points[valid, :]
  target_occluded = target_occluded[valid, :]

  query_points = []
  for i in range(target_points.shape[0]):
    index = np.where(target_occluded[i] == 0)[0][0]
    x, y = target_points[i, index, 0], target_points[i, index, 1]
    query_points.append(np.array([index, y, x]))  # [t, y, x]
  query_points = np.stack(query_points, axis=0)

  return {
      'video': frames[np.newaxis, ...],
      'query_points': query_points[np.newaxis, ...],
      'target_points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np.newaxis, ...],
  }

















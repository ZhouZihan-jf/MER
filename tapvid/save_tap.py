import io
import mmcv
import os
import os.path as osp
import numpy as np
from PIL import Image
import pandas as pd
import json
from datasets.utils.figures import *
from datasets.utils import improc as pips_vis
from tensorboardX import SummaryWriter
import mediapy as media
import random
import colorsys
from glob import glob


def _get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Gets colormap for points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(
            (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        )
    random.shuffle(colors)
    return colors

def generate_video(rgbs, path, fps=20):
    media.write_video(path, rgbs, fps=fps)
    
def add_prefix(inputs, prefix):
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs

def paint_point_track(
        frames: np.ndarray,
        point_tracks: np.ndarray,
        visibles: np.ndarray,
) -> np.ndarray:
    """Converts a sequence of points to color code video.
    Args:
      frames: [num_frames, height, width, 3], np.uint8, [0, 255]
      point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
      visibles: [num_points, num_frames], bool
    Returns:
      video: [num_frames, height, width, 3], np.uint8, [0, 255]
    """
    num_points, num_frames = point_tracks.shape[0:2]
    colormap = _get_colors(num_colors=num_points)
    height, width = frames.shape[1:3]
    dot_size_as_fraction_of_min_edge = 0.015
    radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
    diam = radius * 2 + 1
    quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
    quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
    icon = (quadratic_y + quadratic_x) - (radius ** 2) / 2.0
    sharpness = 0.15
    icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
    icon = 1 - icon[:, :, np.newaxis]
    icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
    icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
    icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
    icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

    video = frames.copy()
    for t in range(num_frames):
        # Pad so that points that extend outside the image frame don't crash us
        image = np.pad(
            video[t],
            [
                (radius + 1, radius + 1),
                (radius + 1, radius + 1),
                (0, 0),
            ],
        )
        for i in range(num_points):
            # The icon is centered at the center of a pixel, but the input coordinates
            # are raster coordinates.  Therefore, to render a point at (1,1) (which
            # lies on the corner between four pixels), we need 1/4 of the icon placed
            # centered on the 0'th row, 0'th column, etc.  We need to subtract
            # 0.5 to make the fractional position come out right.
            x, y = point_tracks[i, t, :] + 0.5
            x = min(max(x, 0.0), width)
            y = min(max(y, 0.0), height)

            if visibles[i, t]:
                x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                x2, y2 = x1 + 1, y1 + 1

                # bilinear interpolation
                patch = (
                        icon1 * (x2 - x) * (y2 - y)
                        + icon2 * (x2 - x) * (y - y1)
                        + icon3 * (x - x1) * (y2 - y)
                        + icon4 * (x - x1) * (y - y1)
                )
                x_ub = x1 + 2 * radius + 2
                y_ub = y1 + 2 * radius + 2
                image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                                                           y1:y_ub, x1:x_ub, :
                                                           ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

            # Remove the pad
            video[t] = image[radius + 1: -radius - 1, radius + 1: -radius - 1].astype(np.uint8)
    return video



def evaluate(results, metrics='tapvid', origin_dir=None, output_dir=None, logger=None):

    metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
    allowed_metrics = ['pck','tapvid']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
    eval_results = dict()
    if mmcv.is_seq_of(results, list):
        num_feats = len(results[0])
        for feat_idx in range(num_feats):
            cur_results = [result[feat_idx] for result in results]
            eval_results.update(
                add_prefix(
                    tapvid_evaluate(cur_results, output_dir, logger),
                    prefix=f'feat_{feat_idx}'))
    else:
        eval_results.update(tapvid_evaluate(results, origin_dir, output_dir, logger))

    return eval_results

def tapvid_evaluate(results, origin_dir, output_dir, vis_traj=True, logger=None):
    samples = glob(osp.join(origin_dir, '*.pkl'))
    summaries = []
    results_list = []
    
    writer = SummaryWriter(os.path.join(output_dir, 'traj'), max_queue=10, flush_secs=60)
    log_freq = 99999
    sw = pips_vis.Summ_writer(  
                            writer=writer,
                            log_freq=log_freq,
                            fps=24,
                            scalar_freq=int(log_freq/2),
                            just_gif=True
                            )
    
    for vid_idx in range(len(results)):
        video_path = samples[vid_idx]
        sample = mmcv.load(video_path)
        
        if isinstance(sample['video'][0], bytes):
            # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            rgbs = np.array([decode(sample['video'][i]) for i in range(sample['video'].shape[0])])
        else:
            rgbs = np.array([sample['video'][i] for i in range(sample['video'].shape[0])])
            
        rgbs = [ mmcv.imresize(rgbs[i], size=(256, 256)) for i in range(rgbs.shape[0])]
        rgbs = np.stack(rgbs, 0)
        
        trajectories_gt, visibilities_gt, trajectories_pred, visibilities_pred, query_points = results[vid_idx]
        num_points = trajectories_gt.shape[2]
        
        # back to 256x256 for inference on TAP-Vid
        trajectories_gt[..., 0] = trajectories_gt[..., 0] * 256 / 256 
        trajectories_gt[..., 1] = trajectories_gt[..., 1] * 256 / 256
        trajectories_pred[..., 0] = trajectories_pred[..., 0] * 256 / 256 
        trajectories_pred[..., 1] = trajectories_pred[..., 1] * 256 / 256
        
        unpacked_results = []
        
        for n in range(num_points):
            unpacked_result = {
                    "idx": f'{vid_idx}_{n}',
                    "iter": vid_idx,
                    "video_idx": 0,
                    "point_idx_in_video": n,
                    "trajectory_gt": trajectories_gt[0, :, n, :].detach().clone().cpu(),
                    "trajectory_pred": trajectories_pred[0, :, n, :].detach().clone().cpu(),
                    "visibility_gt": visibilities_gt[0, :, n].detach().clone().cpu(),
                    "visibility_pred": visibilities_pred[0, :, n].detach().clone().cpu(),
                    "query_point": query_points[0, n, :].detach().clone().cpu(),
                }
            unpacked_results.append(unpacked_result)
        
        summaries_batch = [compute_summary(res, 'first') for res in unpacked_results] # query_mode ?
        summaries += summaries_batch

        summary_df = compute_summary_df(unpacked_results)
        selected_metrics = ["ade_visible", "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy"]
        selected_metrics_shorthand = {
            "ade_visible": "ADE",
            "average_jaccard": "AJ",
            "average_pts_within_thresh": "<D",
            "occlusion_accuracy": "OA",
        }
        print(summary_df[selected_metrics].to_markdown())
        
        video = paint_point_track(rgbs, trajectories_pred[0].transpose(0,1).cpu().numpy(), visibilities_gt[0].transpose(0,1).cpu().numpy())
        
        # generate_gif(video, f'vis/a{vid_idx}.gif')
        if vis_traj:
            generate_video(video, os.path.join(output_dir, f'{vid_idx}.mp4'))
            
            rgbs_input = torch.from_numpy(rgbs)[None].permute(0,1,4,2,3).cuda()
            rgbs_input  = rgbs_input.float() * 1./255 - 0.5
            # gray_rgbs = torch.mean(rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
            
            if vid_idx % 1 == 0:
                vid_path = os.path.join(output_dir,f'{vid_idx}')
                os.makedirs(vid_path, exist_ok=True)
                
                for n in range(num_points):
                    
                    start_idx = 0
                    for i in range(trajectories_gt.shape[1]):
                        if visibilities_gt[0,i,n]:
                            start_idx = i
                            break
                    
                        # color point/line
                    rgbs = sw.summ_traj2ds_on_rgbs(f'{output_dir}/kp{vid_idx}{n}_trajs_e_on_rgbs', trajectories_pred[0:1,start_idx:,n:n+1], rgbs_input[0:1,start_idx:], cmap='spring', linewidth=2, only_return=True)  
                    
                    rgbs = rgbs[0].permute(0,2,3,1).detach().cpu().numpy()
                    
                    generate_video(rgbs, os.path.join(output_dir, f'{vid_idx}_{n}.mp4'))

    
    metadata = {
    "name": 'tapvid_evaluate',
    "model": 'none',
    "dataset": "tapvid_davis",
    "query_mode": 'first',
    }
    
    result = save_results(summaries, results_list, output_dir, 4, metadata)
    
    return result
        


def save_results(summaries, results_list, output_dir, mostly_visible_threshold, metadata):
    # Save summaries as a json file
    dataset = metadata["dataset"]
    os.makedirs(output_dir, exist_ok=True)
    summaries_path = os.path.join(output_dir, f"summaries{dataset}.json")
    with open(summaries_path, "w", encoding="utf8") as f:
        json.dump(summaries, f)
    print(f"\nSummaries saved to:\n{summaries_path}\n")

    # Save results summary dataframe as a csv file
    results_df_path = os.path.join(output_dir, f"results_df{dataset}.csv")
    results_df = pd.DataFrame.from_records(summaries)
    results_df.to_csv(results_df_path)
    print(f"\nResults summary dataframe saved to:\n{results_df_path}\n")
    for k, v in metadata.items():
        results_df[k] = v

    # # Save results summary dataframe as a wandb artifact
    # artifact = wandb.Artifact(name=f"{wandb.run.name}__results_df", type="df", metadata=metadata)
    # artifact.add_file(results_df_path, "results_df.csv")
    # wandb.log_artifact(artifact)

    # Save results list as a pickle file
    if len(results_list) > 0:
        results_list_pkl_path = os.path.join(output_dir, f"results_list{dataset}.pkl")
        with open(results_list_pkl_path, "wb") as f:
            print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
            pickle.dump(results_list, f)

    # Make figures
    figures_dir = os.path.join(output_dir, f"figures{dataset}")
    ensure_dir(figures_dir)
    result = make_figures(results_df, figures_dir, mostly_visible_threshold)

    return result
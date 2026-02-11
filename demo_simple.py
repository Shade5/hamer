"""
Simplified HaMeR Demo - Egocentric hand keypoint extraction

Optimized for egocentric (first-person) views with at most 2 hands.
Outputs 2D keypoints (full image coords) and 3D keypoints (MANO coords).
"""

import warnings
warnings.filterwarnings("ignore", message=".*10 shape coefficients.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention.*")

import torch
import numpy as np
import cv2
import av
from pathlib import Path
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import cam_crop_to_full

OUTPUT_DIR = Path("outputs_simplified")


def extract_hand_bboxes(vitposes_out: list, min_keypoints: int = 3,
                        confidence_thresh: float = 0.5) -> tuple:
    """
    Extract hand bboxes from ViTPose keypoints.
    Returns at most 1 left + 1 right hand (highest confidence).
    """
    best_left = None
    best_left_conf = 0
    best_right = None
    best_right_conf = 0

    for vitposes in vitposes_out:
        keypoints = vitposes['keypoints']

        # Left hand (indices -42 to -21)
        left_hand_keyp = keypoints[-42:-21]
        valid = left_hand_keyp[:, 2] > confidence_thresh
        if valid.sum() > min_keypoints:
            avg_conf = left_hand_keyp[valid, 2].mean()
            if avg_conf > best_left_conf:
                best_left_conf = avg_conf
                best_left = [
                    left_hand_keyp[valid, 0].min(),
                    left_hand_keyp[valid, 1].min(),
                    left_hand_keyp[valid, 0].max(),
                    left_hand_keyp[valid, 1].max()
                ]

        # Right hand (indices -21 to end)
        right_hand_keyp = keypoints[-21:]
        valid = right_hand_keyp[:, 2] > confidence_thresh
        if valid.sum() > min_keypoints:
            avg_conf = right_hand_keyp[valid, 2].mean()
            if avg_conf > best_right_conf:
                best_right_conf = avg_conf
                best_right = [
                    right_hand_keyp[valid, 0].min(),
                    right_hand_keyp[valid, 1].min(),
                    right_hand_keyp[valid, 0].max(),
                    right_hand_keyp[valid, 1].max()
                ]

    bboxes = []
    is_right = []
    if best_left is not None:
        bboxes.append(best_left)
        is_right.append(0)
    if best_right is not None:
        bboxes.append(best_right)
        is_right.append(1)

    if len(bboxes) == 0:
        return None, None

    return np.array(bboxes), np.array(is_right)


def crop_and_preprocess(img_bgr: np.ndarray, boxes: np.ndarray,
                        is_right: np.ndarray, model_cfg,
                        rescale_factor: float = 2.0) -> dict:
    """
    Crop hands and preprocess for HaMeR model.
    Vectorized where possible.
    """
    img_size = model_cfg.MODEL.IMAGE_SIZE
    bbox_shape = model_cfg.MODEL.get('BBOX_SHAPE', None)
    mean = 255.0 * np.array(model_cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(model_cfg.MODEL.IMAGE_STD)

    boxes = boxes.astype(np.float32)
    centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

    batch_imgs = []
    batch_centers = []
    batch_sizes = []

    img_h, img_w = img_bgr.shape[:2]

    for i in range(len(boxes)):
        center = centers[i]
        scale = scales[i]

        # Expand to aspect ratio
        if bbox_shape is not None:
            w, h = scale * 200
            w_t, h_t = bbox_shape
            if h / w < h_t / w_t:
                h_new = max(w * h_t / w_t, h)
                w_new = w
            else:
                h_new = h
                w_new = max(h * w_t / h_t, w)
            bbox_size = max(w_new, h_new)
        else:
            bbox_size = (scale * 200).max()

        flip = (is_right[i] == 0)

        # Prepare image (flip if left hand)
        if flip:
            img = img_bgr[:, ::-1, :]
            cx = img_w - center[0] - 1
        else:
            img = img_bgr
            cx = center[0]
        cy = center[1]

        # Affine transform
        src = np.array([
            [cx, cy],
            [cx, cy + bbox_size * 0.5],
            [cx + bbox_size * 0.5, cy]
        ], dtype=np.float32)

        dst = np.array([
            [img_size * 0.5, img_size * 0.5],
            [img_size * 0.5, img_size],
            [img_size, img_size * 0.5]
        ], dtype=np.float32)

        trans = cv2.getAffineTransform(src, dst)
        img_patch = cv2.warpAffine(img, trans, (img_size, img_size),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

        # BGR -> RGB, HWC -> CHW, normalize
        img_patch = img_patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img_patch = (img_patch - mean[:, None, None]) / std[:, None, None]

        batch_imgs.append(img_patch)
        batch_centers.append(center)
        batch_sizes.append(bbox_size)

    return {
        'img': torch.from_numpy(np.stack(batch_imgs).astype(np.float32)),
        'box_center': torch.from_numpy(np.stack(batch_centers).astype(np.float32)),
        'box_size': torch.from_numpy(np.array(batch_sizes, dtype=np.float32)),
        'img_size': torch.from_numpy(np.array([[img_w, img_h]] * len(boxes), dtype=np.float32)),
        'right': torch.from_numpy(is_right.astype(np.float32)),
    }


def get_keypoints(out: dict, batch: dict, model_cfg) -> dict:
    """
    Extract 2D and 3D keypoints from HaMeR output.
    Reprojects 3D keypoints to 2D using the camera parameters.
    """
    is_right = batch['right']
    box_center = batch['box_center']
    box_size = batch['box_size']
    img_size = batch['img_size']

    pred_keypoints_3d = out['pred_keypoints_3d']
    pred_cam = out['pred_cam'].clone()

    # Flip x component of camera for left hands
    multiplier = 2 * is_right - 1
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]

    # Compute focal length for full image
    focal_length = model_cfg.EXTRA.FOCAL_LENGTH
    crop_size = model_cfg.MODEL.IMAGE_SIZE
    scaled_focal_length = focal_length / crop_size * img_size.max(dim=1).values

    # Convert camera from crop to full image coordinates
    cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

    # Get 3D keypoints and flip x for left hands
    kp_3d = pred_keypoints_3d.clone()
    left_mask = (is_right == 0)
    kp_3d[left_mask, :, 0] = -kp_3d[left_mask, :, 0]

    # Convert 3D keypoints to camera frame
    kp_3d_np = kp_3d.cpu().numpy()
    cam_t_np = cam_t_full.cpu().numpy()
    kp_3d_cam = kp_3d_np + cam_t_np[:, None, :]  # (B, 21, 3) in camera frame

    # Project 3D keypoints to 2D using cv2.projectPoints
    img_size_np = img_size.cpu().numpy()
    focal_np = scaled_focal_length.cpu().numpy()

    assert len(np.unique(focal_np)) == 1, "All images should have exactly 1 focal length."
    rvec = np.zeros((3, 1), dtype=np.float64)  # Identity rotation
    tvec = np.zeros((3, 1), dtype=np.float64)  # Already in camera frame
    f = focal_np[0]
    cx, cy = img_size_np[0] / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    kp_2d_full = []
    for i in range(len(kp_3d_cam)):
        points_2d, _ = cv2.projectPoints(kp_3d_cam[i].astype(np.float64), rvec, tvec, K, distCoeffs=None)
        kp_2d_full.append(points_2d.reshape(-1, 2))

    kp_2d_full = np.stack(kp_2d_full)

    return {
        'keypoints_2d': kp_2d_full,
        'keypoints_3d': kp_3d_cam,
        'is_right': is_right.cpu().numpy(),
    }


def draw_hands(img_rgb: np.ndarray, keypoints: dict) -> np.ndarray:
    """Draw 2D hand keypoints on image."""
    vis_img = img_rgb.copy()
    colors = [(0, 0, 255), (0, 255, 0)]  # Red for left, Green for right

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    for i in range(len(keypoints['keypoints_2d'])):
        color = colors[int(keypoints['is_right'][i])]
        kp = keypoints['keypoints_2d'][i].astype(int)

        for x, y in kp:
            cv2.circle(vis_img, (x, y), 4, color, -1)

        for s, e in connections:
            cv2.line(vis_img, tuple(kp[s]), tuple(kp[e]), color, 2)

    return vis_img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand-detector', type=str, default='vitpose',
                        choices=['vitpose', 'mediapipe'],
                        help='Hand bbox detector to use (default: vitpose)')
    args = parser.parse_args()
    hand_detector_type = args.hand_detector

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading HaMeR model...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    if hand_detector_type == "mediapipe":
        from mediapipe_hand_detector import MediaPipeHandDetector
        print("Loading MediaPipe hand detector...")
        mp_model_path = "/mnt/sunny_nas/weights/hamer_weights/hand_landmarker.task"
        hand_detector = MediaPipeHandDetector(mp_model_path)
    elif hand_detector_type == "vitpose":
        from vitpose_model import ViTPoseModel
        print("Loading ViTPose...")
        vitpose_model = ViTPoseModel(device)

    rescale_factor = 2.0

    # Open video
    video_path = '/mnt/sunny_nas/ai_logs/ailog_RIG-D05_2026_02_02-20_27_04_2f8ea9e2/video.mp4'
    in_container = av.open(video_path, "r")
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"

    start_sec = 0
    in_container.seek(int(start_sec / in_stream.time_base), stream=in_stream,
                      any_frame=False, backward=True)

    # Output video
    out_container = av.open(str(OUTPUT_DIR / "out.mkv"), mode="w")
    out_stream = out_container.add_stream('h264', rate=30)
    out_stream.width = 1024
    out_stream.height = 1024

    print("Processing video...")
    all_keypoints = {}  # pts -> keypoints
    frame_idx = 0

    for frame in tqdm(in_container.decode(in_stream), total=out_stream.frames):
        img = frame.to_ndarray(format="rgb24")
        img_rgb = img[:1024]  # Left half
        img_bgr = img_rgb[:, :, ::-1].copy()

        # Hand detection
        if hand_detector_type == "mediapipe":
            timestamp_ms = int(frame_idx * 1000 / 30)
            boxes, is_right = hand_detector.detect(img_rgb, timestamp_ms)
        else:
            h, w = img_rgb.shape[:2]
            vitposes_out = vitpose_model.predict_pose(img_rgb, [np.array([[0, 0, w, h, 1.0]])])
            boxes, is_right = extract_hand_bboxes(vitposes_out)
        frame_idx += 1
        if boxes is None:
            all_keypoints[frame.pts] = None
            out_frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)
            continue

        # Crop and preprocess
        batch = crop_and_preprocess(img_bgr, boxes, is_right, model_cfg, rescale_factor)
        batch = recursive_to(batch, device)

        # HaMeR inference
        with torch.no_grad():
            out = model(batch)

        # Get keypoints (reproject 3D to 2D)
        keypoints = get_keypoints(out, batch, model_cfg)
        all_keypoints[frame.pts] = keypoints

        # Draw and encode
        vis_img = draw_hands(img_rgb, keypoints)
        out_frame = av.VideoFrame.from_ndarray(vis_img, format="rgb24")
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

    # Flush encoder
    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()
    if hand_detector_type == "mediapipe":
        hand_detector.close()
    print(f"Saved to {OUTPUT_DIR / 'out.mkv'}")

    # Save keypoints
    keypoints_path = Path(video_path).parent / "hamer_keypoints.npz"
    np.savez(keypoints_path, keypoints=all_keypoints)
    print(f"Saved keypoints to {keypoints_path}")


if __name__ == '__main__':
    main()

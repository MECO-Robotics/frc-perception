#!/usr/bin/env python3
"""Build a synthetic mono replay scene with 100 occluding FUEL balls.

Ball pixels come from the saved crop replay; this is a detector/load fixture,
not calibrated stereo ground truth.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = ROOT / "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"
OUTPUT = ROOT / "media/frc-synthetic-100-fuel-motion-60fps.mp4"
STEREO_OUTPUT = ROOT / "media/frc-synthetic-stereo-100-fuel-motion-60fps.mp4"
SEED = 20261105
WIDTH, HEIGHT = 1600, 1200
FPS, DURATION_S = 60.0, 10.0
BALL_COUNT = 100


def extract_sprite(frame, center, radius=51):
    cx, cy = center
    x0, y0 = int(cx - radius - 4), int(cy - radius - 4)
    size = 2 * radius + 8
    patch = frame[y0:y0 + size, x0:x0 + size].copy()
    if patch.shape[:2] != (size, size):
        raise ValueError("Sprite falls outside source frame")
    yy, xx = np.mgrid[:size, :size]
    distance = np.sqrt((xx - (size - 1) / 2) ** 2 +
                       (yy - (size - 1) / 2) ** 2)
    # A slight feather hides the hard cut while retaining the real sphere edge.
    alpha = np.clip((radius + 1.5 - distance) / 2.5, 0, 1)
    return patch, alpha.astype(np.float32)


def make_background(crop):
    # Reuse the dark room and fabric texture from the real crop, without copying
    # either ball into the synthetic scene.
    top_source = crop[0:75, :, :].astype(np.float32)
    top_color = np.median(top_source.reshape(-1, 3), axis=0)
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    top = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    # Dim background above the fabric plane.
    base = top_color[None, None, :] * (0.7 + 0.5 * y)
    top[:] = base

    # The source crop's bottom strip is clean fabric. Stretch it into a floor
    # plane so there are no tile seams or ball pixels in the background.
    strip = crop[218:256, :, :]
    floor_start = 280
    floor = cv2.resize(strip, (WIDTH, HEIGHT - floor_start),
                       interpolation=cv2.INTER_CUBIC).astype(np.float32)
    floor = cv2.GaussianBlur(floor, (0, 0), 1.3)
    floor_y = np.linspace(0, 1, HEIGHT - floor_start,
                          dtype=np.float32)[:, None, None]
    floor *= 0.72 + 0.28 * floor_y
    top[floor_start:] = floor
    blend_height = 100
    for offset in range(blend_height):
        weight = offset / blend_height
        row = floor_start - blend_height + offset
        top[row] = top[row] * (1 - weight) + floor[0] * weight
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 1.8, (HEIGHT, WIDTH, 1)).astype(np.float32)
    return np.clip(top + noise, 0, 255).astype(np.uint8)


def placements():
    rng = random.Random(SEED)
    balls = []
    # Ten staggered rows of ten. Balls get larger toward the foreground and
    # neighboring rows overlap enough to hide substantial parts of many balls.
    for row in range(10):
        diameter = 54 + row * 7.5
        y = 245 + row * 80
        offset = (44 if row % 2 else 0)
        for col in range(10):
            x = 300 + col * 106 + offset
            x += rng.uniform(-9, 9)
            cy = y + rng.uniform(-7, 7)
            balls.append({
                "id": row * 10 + col,
                "x": x,
                "y": cy,
                "diameter": diameter * rng.uniform(0.93, 1.07),
                "row": row,
                "brightness": rng.uniform(0.88, 1.12),
                "sprite": rng.randrange(2),
                "phase_x": rng.uniform(0, 2 * math.pi),
                "phase_y": rng.uniform(0, 2 * math.pi),
                "phase_texture": rng.uniform(0, 2 * math.pi),
                "motion_x_px": rng.uniform(22, 58),
                "motion_y_px": rng.uniform(7, 20),
                "omega_x": rng.uniform(0.45, 1.0),
                "omega_y": rng.uniform(0.35, 0.8),
                "texture_angle_deg": rng.uniform(0, 360),
            })
    # Paint far rows first so foreground balls occlude the back rows.
    return sorted(balls, key=lambda ball: (ball["row"], ball["x"]))


def prepare_ball(sprite, alpha, diameter, angle_deg):
    size = max(12, int(round(diameter)))
    resized = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_AREA
                         if size < sprite.shape[0] else cv2.INTER_CUBIC)
    mask = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_LINEAR)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
        resized = cv2.warpAffine(resized, matrix, (size, size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    radius = size // 2
    shadow = np.zeros((max(12, int(size * 0.55)),
                       max(12, int(size * 1.6))), dtype=np.uint8)
    cv2.ellipse(shadow, (shadow.shape[1] // 2, shadow.shape[0] // 2),
                (max(2, int(radius * 0.84)), max(2, int(radius * 0.20))),
                0, 0, 360, 90, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(2, size * 0.08))
    return resized, mask, shadow


def composite_ball(image, prepared, x, y, brightness):
    resized, mask, shadow = prepared
    size = resized.shape[0]
    radius = size // 2
    # Apply a precomputed local shadow, avoiding a full-frame allocation per ball.
    sh_x0 = int(round(x - shadow.shape[1] / 2))
    sh_y0 = int(round(y + radius * 0.60 - shadow.shape[0] / 2))
    x1, y1 = max(0, sh_x0), max(0, sh_y0)
    x2 = min(image.shape[1], sh_x0 + shadow.shape[1])
    y2 = min(image.shape[0], sh_y0 + shadow.shape[0])
    if x1 < x2 and y1 < y2:
        sh = shadow[y1 - sh_y0:y2 - sh_y0, x1 - sh_x0:x2 - sh_x0]
        region = image[y1:y2, x1:x2].astype(np.float32)
        factor = 1 - sh[:, :, None].astype(np.float32) / 255 * 0.48
        image[y1:y2, x1:x2] = (region * factor).astype(np.uint8)

    x0, y0 = int(round(x - radius)), int(round(y - radius))
    bx1, by1 = max(0, x0), max(0, y0)
    bx2, by2 = min(image.shape[1], x0 + size), min(image.shape[0], y0 + size)
    if bx1 >= bx2 or by1 >= by2:
        return
    sx1, sy1 = bx1 - x0, by1 - y0
    sx2, sy2 = sx1 + bx2 - bx1, sy1 + by2 - by1
    patch = image[by1:by2, bx1:bx2].astype(np.float32)
    foreground = np.clip(resized[sy1:sy2, sx1:sx2].astype(np.float32) * brightness,
                         0, 255)
    a = mask[sy1:sy2, sx1:sx2, None]
    image[by1:by2, bx1:bx2] = np.clip(foreground * a + patch * (1 - a),
                                  0, 255).astype(np.uint8)


def render_view(background, balls, time_s, right_eye=False):
    view = background.copy()
    for ball in balls:
        dx = ball["motion_x_px"] * math.sin(
            time_s * ball["omega_x"] + ball["phase_x"])
        dy = ball["motion_y_px"] * math.sin(
            time_s * ball["omega_y"] + ball["phase_y"])
        # Row disparity keeps left/right views paired; it is not metric 3D GT.
        disparity = 8.0 + ball["row"] * (28.0 / 9.0)
        x = ball["x"] + dx - (disparity if right_eye else 0.0)
        y = ball["y"] + dy
        composite_ball(view, ball["prepared"], x, y, ball["brightness"])
    return view


def main():
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    ok, stereo = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read source video: {SOURCE_VIDEO}")
    left = stereo[:, :WIDTH]
    # Pixel locations are from the two detector boxes in the source sequence.
    sprites = [extract_sprite(left, (807, 584)),
               extract_sprite(left, (902, 589))]
    crop = cv2.VideoCapture(str(ROOT / "media/frc-fuel-ball-crop-60fps.mp4"))
    crop_ok, crop_frame = crop.read()
    crop.release()
    if crop_ok:
        background = make_background(crop_frame)
    else:
        background = np.full((HEIGHT, WIDTH, 3), (25, 25, 20), dtype=np.uint8)
    balls = placements()
    for ball in balls:
        sprite, alpha = sprites[ball["sprite"]]
        ball["prepared"] = prepare_ball(
            sprite, alpha, ball["diameter"], ball["texture_angle_deg"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"avc1"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {OUTPUT}")
    stereo_writer = cv2.VideoWriter(
        str(STEREO_OUTPUT), cv2.VideoWriter_fourcc(*"avc1"), FPS,
        (WIDTH * 2, HEIGHT))
    if not stereo_writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create stereo output video: {STEREO_OUTPUT}")
    frame_count = int(round(FPS * DURATION_S))
    for frame_index in range(frame_count):
        t = frame_index / FPS
        left_frame = render_view(background, balls, t, right_eye=False)
        right_frame = render_view(background, balls, t, right_eye=True)
        writer.write(left_frame)
        stereo_writer.write(np.hstack((left_frame, right_frame)))
    writer.release()
    stereo_writer.release()

    metadata = {
        "video": str(OUTPUT.relative_to(ROOT)),
        "source_ball_crop_video": "media/frc-fuel-ball-crop-60fps.mp4",
        "dimensions": [WIDTH, HEIGHT],
        "fps": FPS,
        "duration_s": frame_count / FPS,
        "frames": frame_count,
        "placed_ball_count": BALL_COUNT,
        "layout": "10 staggered rows x 10 balls; far-to-near compositing creates partial occlusion",
        "motion": "all 100 balls move independently on smooth bounded horizontal and vertical trajectories, changing occlusion over time",
        "motion_model": "deterministic sinusoidal image-plane motion; no rigid-body collision solver",
        "purpose": "detector and tracking load/replay fixture",
        "limitations": [
            "mono left-camera view only; no right-camera stereo partner",
            "not calibrated stereo or physical 3D ground truth",
        "some balls are substantially occluded by construction",
            "sprites reuse two source ball textures with modest brightness changes",
        ],
        "seed": SEED,
    }
    meta_path = OUTPUT.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    stereo_metadata = dict(metadata)
    stereo_metadata.update({
        "video": str(STEREO_OUTPUT.relative_to(ROOT)),
        "dimensions": [WIDTH * 2, HEIGHT],
        "layout": "rectified side-by-side synthetic stereo: left view then right view",
        "right_view": "same per-ball motion and sprite pixels as left view, with each ball shifted left by 8-36 px according to row",
        "limitations": [
            "horizontal disparity is assigned by image row, not generated from calibrated 3D points",
            "the background has zero disparity and camera lens distortion is not simulated",
            "not suitable for metric range accuracy or physical stereo validation",
            "sprites reuse two source ball textures with modest brightness changes",
        ],
        "stereo_limitations": [
            "disparity is an image-space simulation, not derived from calibrated 3D positions",
            "background has zero disparity and the camera models/distortion are not simulated",
            "not suitable for measuring metric range accuracy",
        ],
    })
    stereo_meta_path = STEREO_OUTPUT.with_suffix(".json")
    stereo_meta_path.write_text(json.dumps(stereo_metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("stereo video", STEREO_OUTPUT)
    print("stereo metadata", stereo_meta_path)


if __name__ == "__main__":
    main()

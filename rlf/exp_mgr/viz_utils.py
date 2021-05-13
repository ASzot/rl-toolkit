import cv2
import numpy as np
import os.path as osp
import os

def append_text_to_image(image, lines):
    """
    Parameters:
        image: (np.array): The frame to add the text to.
        lines (list):
    Returns:
        image: (np.array): The modified image with the text appended.
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    final = image + blank_image
    return final


def save_agent_obs(frames, imdim, vid_dir, name):
    use_dir = osp.join(vid_dir, name+'_frames')
    if not osp.exists(use_dir):
        os.makedirs(use_dir)

    if imdim != 1:
        raise ValueError('Only gray scale is supported right now')

    for i in range(frames.shape[0]):
        for frame_j in range(frames.shape[1]):
            fname = f"{i}_{frame_j}.jpg"
            frame = frames[i,frame_j].cpu().numpy()
            cv2.imwrite(osp.join(use_dir, fname), frame)

    print(f"Wrote observation sequence to {use_dir}")


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False,
        should_print=True):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + '.mp4')
    if osp.exists(vid_file):
        os.remove(vid_file)

    w, h = frames[0].shape[:-1]
    videodims = (h, w)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(vid_file, fourcc, fps, videodims)
    for frame in frames:
        frame = frame[..., 0:3][..., ::-1]
        video.write(frame)
    video.release()
    if should_print:
        print(f"Rendered to {vid_file}")



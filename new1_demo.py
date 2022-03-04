import matplotlib

matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
import skimage
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='checkpoint/vox-cpk.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='image/obama.jpg', help="path to source image")
    parser.add_argument("--driving_video", default='cropvideo/crop2_002.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=False)
    # parser.set_defaults(find_best_frame=True)
    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    cap=cv2.VideoCapture(0)
    frame_list=[]
    flag=0

    while True:
        # frame_list = []
        ref, frame = cap.read()
        cv2.rectangle(frame, (150, 80), (550, 450), (0,255,0), 2)
        cv2.imshow("1", frame)
        if flag==0:
            frame_copy = frame
            frame_list.append(frame)
            frame_list.append(frame)
            cv2.imshow("frame0", frame_list[0])
            cv2.imshow("frame1", frame_list[1])
            flag=1
            continue
        else:
            frame_list=[]
            frame_list.append(frame_copy)
            frame_list.append(frame)
            cv2.imshow("frame0", frame_list[0])
            frame_list = [frame[..., ::-1] for frame in frame_list]
            frame_list = [frame[80:450, 150:550] for frame in frame_list]
            frame_list = [resize(frame, (256, 256))[..., :3] for frame in frame_list]

            predictions = make_animation(source_image, frame_list, generator, kp_detector, relative=opt.relative,
                                         adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

            predictions = [frame[..., ::-1] for frame in predictions]
            cv2.imshow("result", predictions[-1])



        c = cv2.waitKey(30) & 0xff
        if c == 27:
            cap.release()
            break

    # frame_list = [frame[..., ::-1] for frame in frame_list]
    # frame_list = [frame[80:450, 150:550] for frame in frame_list]
    # frame_list = [resize(frame, (256, 256))[..., :3] for frame in frame_list]

    # if opt.find_best_frame or opt.best_frame is not None:
    #     i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, frame_list,
    #                                                                           cpu=opt.cpu)
    #     print("Best frame: " + str(i))
    #     driving_forward = frame_list[i:]
    #     driving_backward = frame_list[:(i + 1)][::-1]
    #     predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector,
    #                                          relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
    #                                          cpu=opt.cpu)
    #     predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector,
    #                                           relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
    #                                           cpu=opt.cpu)
    #     predictions = predictions_backward[::-1] + predictions_forward[1:]
    # else:
    #     predictions = make_animation(source_image, frame_list, generator, kp_detector, relative=opt.relative,
    #                                  adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

    # predictions = make_animation(source_image, frame_list, generator, kp_detector, relative=opt.relative,
    #                              adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    #
    # imageio.mimsave("i1.mp4", [img_as_ubyte(frame) for frame in frame_list], fps=fps)
    # imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

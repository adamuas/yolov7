import time
from pathlib import Path

import requests
import base64
import torch
import numpy as np
from io import BytesIO
import torch.backends.cudnn as cudnn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from models.experimental import attempt_load
from utils.datasets import (
    letterbox,
    LoadStreams, 
    LoadImages
)
from utils.general import (
    check_img_size, 
    non_max_suppression,
    apply_classifier, 
    scale_coords, 
    xyxy2xywh,
    set_logging,
)

from filestack import Client


from utils.plots import plot_one_box, plot_one_box_with_pillow
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


app = FastAPI()

origins = [
    "http://synapse-ai-platform-dev.azurewebsites.net/",
    "https://synapse-ai-platform-dev.azurewebsites.net/",
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_class_colorcode(class_name:str) -> tuple:
    """ 
    Returns Colour code  for the class name given 
    """
    VEHICLE_CODE = [255, 153, 0]
    default_colour_codes = {
        'person': [ 102, 0, 153 ],
        'car': VEHICLE_CODE,
        'truck': VEHICLE_CODE,
        'airplane': VEHICLE_CODE,
        'motorcycle': VEHICLE_CODE
    }

    color_code = default_colour_codes.get(class_name, None)

    if color_code is None:
        return  [ 255, 156, 0]
    else:
        return color_code


def upload_to_filestack(img):
    """upload to filestack"""
    filestack_client = Client("A3NlCDjjSR5OprMnZyPvgz")
    img.save('predictions.png', format='PNG')
    #  Upload
    store_params = {
    "mimetype": "image/png"
    }

    new_filelink = filestack_client.upload(filepath="predictions.png",store_params=store_params)

    return new_filelink.url

def detect(image, img0, entities_expected = ['person', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                                                           'truck', 'boat', 'backpack', 'handbag', 'suitcase', 'knife', 
                                                           'laptop', 'cell phone', 'keyboard', 'book', 'clock'], ):
    # setup optional params
    source, weights, imgsz, trace = (
        "inference/images", 
        "yolov7.pt", 
        640,
        not True
    )
   
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print(names)
    colors = [get_class_colorcode(name_i) for name_i in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    

    for path, img, im0s, vid_cap in [(None, image, img0, None)]:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        lst_detections = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] not in entities_expected:
                        # TODO at a later - makes this probablistic (entities expected * confidence)
                        continue
                    
                    # generate xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # plot with pillow
                    conf_percent = conf * 100
                    label = f' {names[int(cls)]} ({conf_percent:.2f}%) '
                    im0 = plot_one_box_with_pillow(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    lst_detections.append({"xywh": xywh, "conf_percent": conf_percent, "label": label})

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    annotated_img = Image.fromarray(im0)
    return {"detections": lst_detections}, annotated_img



"""
Endpoints 
"""
@app.get("/health")
def health_check():
    return {"response": "pongaroo!"}

@app.post("/analyse")
def detect_objects(image_url:str):
    print(image_url)
    with requests.get(image_url) as res:
        if res.status_code == 200:
            img_org = Image.open(BytesIO(res.content))
            img = letterbox(np.array(img_org), 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
         
            detections, annotated_img = detect(img, np.array(img_org))
            annotated_filelink = upload_to_filestack(annotated_img)
            print(annotated_filelink)
        else:
            return {"error_message": "I was unable to read the image"}


    return {**detections,"annotatedFileLink": annotated_filelink}

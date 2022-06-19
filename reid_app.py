import os
import time
import tkinter as tk
from collections import defaultdict

import numpy as np
import cv2
import PIL.Image, PIL.ImageTk

import torch
from torchreid.utils import FeatureExtractor
from torchreid import metrics

REID_REGISTRY = defaultdict(dict)

class YOLO:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        self.model.cuda()       
    
    def detect(self, img, video_source):
        REID_REGISTRY[video_source]['original_img'] = img
        REID_REGISTRY[video_source]['crops'] = []
        REID_REGISTRY[video_source]['bboxes'] = []

        result = self.model(img) # yolo detections
        df = result.pandas().xyxy[0] # bboxes 
        df = df[(df.name == 'person') & (df.confidence > 0.6)].values[:, :4].astype(np.uint)

        for xmin, ymin, xmax, ymax in df: 
            # save crop people
            REID_REGISTRY[video_source]["crops"].append(img[ymin:ymax, xmin:xmax, :])
            REID_REGISTRY[video_source]["bboxes"].append([xmin, ymin, xmax, ymax])
            

YOLO_MODEL = YOLO()

class REID:
    def __init__(self) -> None:
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights\\torchreid\\osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
            image_size=(256, 128),
            pixel_mean= [0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
        )    
        self.color_list = [(128, 0, 0), (0, 128, 0), (0, 0, 128), (0, 0, 0)]
        
          
    def extract_features(self):
    
        self.query_features = self.extractor(REID_REGISTRY['query']['crops'])
        self.gallery_features = self.extractor(REID_REGISTRY['gallery']['crops'])

        distmat = metrics.compute_distance_matrix(self.query_features, self.gallery_features)
        self.distmat = distmat.cpu().numpy()
        
    
    def matching(self):
        # matching
        self.matches = []

        query_bboxes = REID_REGISTRY['query']['bboxes']
        gallery_bboxes = REID_REGISTRY['gallery']['bboxes']

        gallery_remainders = []
        
        while len(query_bboxes) and len(gallery_bboxes) != len(gallery_remainders):
            ymin, xmin = np.unravel_index(self.distmat.argmin(), self.distmat.shape)
            if not xmin in gallery_remainders:
                self.matches.append((query_bboxes.pop(ymin), gallery_bboxes[xmin]))
                self.distmat = np.delete(self.distmat, ymin, axis=0) 
                gallery_remainders.append(xmin)
            else:                   
                self.distmat[ymin, xmin] = np.inf
        
        self.gallery_remainers = [i for i in range(len(gallery_bboxes)) if not i in gallery_remainders]
        
    
    def render(self):
        query_img = REID_REGISTRY['query']['original_img'] 
        gallery_img = REID_REGISTRY['gallery']['original_img'] 

        for i, (q, g) in enumerate(self.matches):
            query_img = cv2.rectangle(query_img, (q[0], q[1]), (q[2], q[3]), color=self.color_list[i], thickness=2) 
            query_img = cv2.putText(query_img, str(i), (q[0], q[1]-2), 0, 0.7, self.color_list[i], 1)
            gallery_img = cv2.rectangle(gallery_img, (g[0], g[1]), (g[2], g[3]), color=self.color_list[i], thickness=2)
            gallery_img = cv2.putText(gallery_img, str(i), (g[0], g[1]-2), 0, 0.7, self.color_list[i], 1) 

        gallery_bboxes = REID_REGISTRY['gallery']['bboxes'] # remainers
        for g in self.gallery_remainers:
            g_b = gallery_bboxes[g]
            gallery_img = cv2.rectangle(gallery_img, (g_b[0], g_b[1]), (g_b[2], g_b[3]), color=self.color_list[-1], thickness=2) 
            gallery_img = cv2.putText(gallery_img, 'Hello!', (g_b[0], g_b[1]-2), 0, 0.7, self.color_list[-1], 1) 
        
        query_bboxes = REID_REGISTRY['query']['bboxes'] # remainers
        for q in query_bboxes:
            query_img = cv2.rectangle(query_img, (q[0], q[1]), (q[2], q[3]), color=self.color_list[-1], thickness=2) 
            query_img = cv2.putText(query_img, 'Hello!', (q[0], q[1]-2), 0, 0.7, self.color_list[-1], 1) 

        return query_img, gallery_img
            

REID_MODEL = REID()

class App:
    def __init__(self, window, window_title, videos_path):
        self.window = window
        self.window.title(window_title)

        # videos
        videos = [os.path.join(videos_path, i) for i in os.listdir(videos_path)]

        # open video source (by default this will try to open the computer webcam)
        self.vid_1 = MyVideoCapture(videos[0])
        self.vid_2 = MyVideoCapture(videos[1])

        # Create a canvas that can fit the above video source size

        self.canvas = tk.Canvas(window, width = 500, height = 620)
        self.canvas.pack()

        self.fps_label_text = tk.StringVar()
        self.fps_label_text.set("")
        self.fps_label = tk.Label(textvariable=self.fps_label_text, justify=tk.CENTER, fg="#eee", bg="#333")
        self.fps_label.place(relx=.2, rely=.3)
        self.fps_label.pack()

        # # Button that lets the user take a snapshot
        # self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        # self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 3

        self.update()

        self.window.mainloop()
 
    # def snapshot(self):
    #     # Get a frame from the video source
    #     ret, frame = self.vid.get_frame()

    #     if ret:
    #         cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        start = time.time()

        frame_1 = self.vid_1.get_frame()
        frame_2 = self.vid_2.get_frame()

        # detect people on both cameras
        YOLO_MODEL.detect(frame_1, 'query')
        YOLO_MODEL.detect(frame_2, 'gallery')

        # matching and render
        if len(REID_REGISTRY['query']['crops']) and len(REID_REGISTRY['gallery']['crops']):
            REID_MODEL.extract_features()
            REID_MODEL.matching()
            
            frame_1, frame_2 = REID_MODEL.render()
        
        self.photo_1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_1))
        self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))

        self.canvas.create_image(0, 0, image = self.photo_1, anchor = tk.NW)
        self.canvas.create_image(0, 310, image = self.photo_2, anchor = tk.NW)

        end = time.time()

        fps = 2 / (end - start)
        self.fps_label_text.set("FPS: {}".format(np.round(fps,0)))

        self.window.after(self.delay, self.update)
        
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (500, 300))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                return frame
            return None
        else:
            return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        
# Create a window and pass it to the Application object

App(tk.Tk(), "Test task Visionero", 'video_data')


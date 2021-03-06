# Two streams ReID

### Pipeline

cv2 (read images) -> YOLOv5 (detects people, saves crops, saves bboxes) -> torchreid (ReID task) -> tkinter (show result)

### Ideas 
* Use sort or deep sort for reid in one frame
* Use docker


### Instalation

```
conda create --name reid python=3.7
conda activate reid

pip install -r yolo_requirements.py

git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/

pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

python setup.py develop
```

### Get started

```
python reid_app.py
```

### Video results: [google.drive](https://drive.google.com/drive/folders/1KmIl_g3aaXqmw_a8meM83dVVk8mAv9z-?usp=sharing)
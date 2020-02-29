# VTouchDFDC

## Setup

### Docker
- Get docker image
```bash
docker pull poperson1205/vtouch-dfdc:latest
```
- Run option
```bash
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --gpus all --name vtouch-dfdc poperson1205/vtouch-dfdc:latest
```

### Dataset
- Cropped faces (224x224)
<https://www.kaggle.com/dagnelies/deepfake-faces>

## Progress
- [x] Submit randomly generated estimations --> (score: 1.00441, rank: 1311/1457)
- [x] Train binary classifier
- [x] Balance positive/negative examples
- [x] Normalize input images
- [x] Check multiple frames for each video --> (score: 0.62684, rank: 865/1928)
- [ ] Refine training data

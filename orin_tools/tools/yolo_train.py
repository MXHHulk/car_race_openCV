#!/usr/bin/env python3
"""訓練車道線 YOLO 分割模型（資料集由 tools/yolo_dataset.py 產生）。

  python3 tools/yolo_train.py                      # 預設 YOLOv8n-seg、320、25 輪
  python3 tools/yolo_train.py --device 0           # 有 CUDA 版 torch 時用顯卡

輸出 runs/lane/train*/weights/best.pt，複製到 lane/models/lane_seg.pt 給 vision.py 用。
Orin 上轉成 TensorRT 比較快：yolo export model=lane_seg.pt format=engine imgsz=320 half=True
"""
import argparse
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "..", "yolo_lane", "data.yaml"))
    ap.add_argument("--model", default="yolov8n-seg.pt", help="預訓練權重（第一次會下載）")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model = YOLO(args.model)
    model.train(data=os.path.abspath(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=4, project=os.path.join(ROOT, "..", "runs", "lane"), name="train",
                # 車道線不能左右翻轉以外的變形太大：保留亮度變化（隧道、反光），關掉馬賽克
                fliplr=0.5, mosaic=0.0, degrees=0.0, scale=0.2, hsv_v=0.5, plots=False, verbose=False)


if __name__ == "__main__":
    main()

# export_onnx.py
import torch
import argparse
import os

def export(ckpt_path, onnx_out, img_size=224):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get('classes', None)
    # Build model architecture (must match train)
    from model import create_model
    num_classes = len(classes) if classes else 4
    model = create_model(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dummy = torch.randn(1,3,img_size,img_size)
    torch.onnx.export(model, dummy, onnx_out, opset_version=13,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}})
    print("ONNX exported:", onnx_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.img_size)

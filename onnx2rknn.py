from rknn.api import RKNN


def generate_rknn():
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")

    if rknn.load_onnx(model=f'models/yolov8n.onnx') != 0:
        print('ONNX model loading failed')

    if rknn.build(do_quantization=True, dataset="rknn-data/dataset.txt") != 0:
        print("Build failed")

    if rknn.export_rknn(f"models/yolov8n.rknn") != 0:
        print("RKNN model export failed")

    rknn.release()

    print("Successfully exported RKNN model")


if __name__ == '__main__':
    generate_rknn()

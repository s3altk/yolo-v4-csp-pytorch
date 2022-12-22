import argparse
import torch
import numpy as np
import warnings

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=608, help='image size')
    
    opt = parser.parse_args()
    print(opt)
    
    # Initialize model with the pretrained weights
    # Set the model to inference mode
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model.export = True  # set Detect() layer export=True

    # Input to the model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    img = torch.randn(1, 3, imgsz, imgsz, requires_grad=True)
    out = model(img)
    shape = tuple((out[0] if isinstance(out, tuple) else out).shape)  # model output shape
    print(f'\nModel with output shape {shape}...')

    # ONNX export
    try:
        import onnx
        import onnxruntime

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        
        f = opt.weights.replace('.pt', '.onnx')  # filename
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}
        
        warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        
        # Export the model
        torch.onnx.export(model,                     # model being run
                  img,                               # model input (or a tuple for multiple inputs)
                  f,                                 # where to save the model (can be a file or file-like object)
                  verbose=False,                     # store the trained parameter weights inside the model file
                  opset_version=12,                  # the ONNX version to export the model to
                  do_constant_folding=True,          # whether to execute constant folding for optimization
                  input_names = ['images'],          # the model's input names
                  output_names = ['output0'],        # the model's output names
                  dynamic_axes = dynamic)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        
        ort_session = onnxruntime.InferenceSession(f)
        
        # Compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: img.detach().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        print(f'PyTorch: {out[0].detach().numpy()}')
        print(f'ONNX: {ort_outs[0]}')
        
        # Compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(out[0].detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
        print('Exported model has been tested with ONNXRuntime and the result looks good!')
        
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
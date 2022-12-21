import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[608, 608], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    
    opt = parser.parse_args()
    print(opt)

    # Initialize model with the pretrained weights
    torch_model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()

    # Set the model to inference mode
    torch_model.eval()
    torch_model.model[-1].export = True  # set Detect() layer export=True

    # Input to the model
    x = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size (1,3,608,608)
    torch_out = torch_model(x)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        
        # Export the model
        torch.onnx.export(torch_model,               # model being run
                  x,                                 # model input (or a tuple for multiple inputs)
                  f,                                 # where to save the model (can be a file or file-like object)
                  export_params=True,                # store the trained parameter weights inside the model file
                  opset_version=11,                  # the ONNX version to export the model to
                  do_constant_folding=True,          # whether to execute constant folding for optimization
                  input_names = ['input'],           # the model's input names
                  output_names = ['output'],         # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
import onnx
from pathlib import Path
from tools.misc import ModelSource
import torch
from nanosam2.sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnxsim

def modify_filename(file_path, str_extension="_modified") -> Path:
    """
    Add the string str_extension before the file extension to the name of a file.
    """
    # Check if the input is already a Path object
    if not isinstance(file_path, Path):
        # Create a Path object from the file path string
        path = Path(file_path)
    else:
        path = file_path

    # Create the new file name by adding "modified" before the extension
    new_file_name = path.with_name(f"{path.stem}{str_extension}{path.suffix}")

    return new_file_name

def remove_duplicate_outputs(model_path:str, out_file:str=None) -> onnx.ModelProto:
    """
    Remove duplicate output nodes from an onnx model.
    Nodes are considered as duplicated if they have the same name and output shape.
    """
    model = onnx.load(model_path)

    # Get the current output nodes
    original_outputs = model.graph.output

    # Dictionary to track unique outputs
    unique_outputs = {}
    new_outputs = []
    to_remove = []

    # Iterate through the output nodes
    for output in original_outputs:
        # Create a key based on the output name and shape
        shape = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
        key = (output.name, shape)

        # Check if this key already exists
        if key not in unique_outputs:
            unique_outputs[key] = output
            new_outputs.append(output)
        else:
            print(f"Duplicate output found: {output.name} with shape: {shape}. Removing it.")
            to_remove.append(output)

    # Update the model's output nodes
    for o in to_remove:
        model.graph.output.remove(o)

    if out_file is not None:
        onnx.save(model, out_file)
        print(f"Modified model saved to {out_file}")
    return {"onnx_model":model, "out_file": out_file}


def analyze_onnx_model(model:onnx.ModelProto):
    """
    Identify the in- and output nodes of a onnx model.
    """
    graph = model.graph

    for input in graph.input:
        output_name = input.name
        output_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f'Input Name: {output_name}, Shape: {output_shape}')

    for output in graph.output:
        output_name = output.name
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f'Output Name: {output_name}, Shape: {output_shape}')

def test_onnx_model(model_path:Path, input):
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input})

    print("ONNX model output shapes:")
    for i, output in enumerate(outputs):
        print(f"{i} | shape: {output.shape}")
    return output

def determine_torch_output_shapes(y, id=0):
    """
    Determine the output shapes of the feature map produced by a torch inference.
    """
    if isinstance(y, dict):
        for k,v in y.items():
            if isinstance(list, v) or isinstance(tuple, v):
                determine_torch_output_shapes(v)
            else:
                print(f"{id} | {k} shape: {v.shape}")
                id += 1
    elif isinstance(y, list) or isinstance(y, tuple):
        for k,v in enumerate(y):
            determine_torch_output_shapes(v)
    else:
        print(f"out | - {y.shape}")

def test_torch_model(torch_model:torch.nn, input, silent=False):
    y = torch_model(input)
    if not silent:
        determine_torch_output_shapes(y)

def export_model_block(m:ModelSource, block:str, out_dir:Path, use_simplify:bool=False):
    """
    Export a building block of nanosam2 to onnx. Not all blocks can be converted.
    """
    print(f"Exporting Model: {m.name} block: {block}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(m.cfg, m.checkpoint, torch.device("cpu"))
    separate_parameters = False
    additional_parameters = None
    match block:
        case "image-encoder":
            torch_model = predictor.image_encoder
            inputs = [[1,3,512,512]]
        case "image-encoder-trunk":
            torch_model = predictor.image_encoder.trunk
            inputs = [[1,3,512,512]]
        case "image-encoder-neck":
            torch_model = predictor.image_encoder.neck
            inputs = [[1,64,128,128],
                      [1,128,64,64],
                      [1,256,32,32],
                      [1,512,16,16]]
        case "mask-decoder-transformer":
            torch_model = predictor.sam_mask_decoder.transformer
            inputs = [[1, 256, 32, 32],
                      [1, 256, 32, 32],
                      [1, 8, 256]]
            separate_parameters = True
        case "memory-encoder":
            torch_model = predictor.memory_encoder
            inputs = [[1, 256, 32, 32],
                      [1, 1, 512, 512]]
            additional_parameters = [True]
            separate_parameters = True
        case "not supported: prompt-encoder-mask-downscaling":
            # mask_downscaling is only used in SAM2 when prompting with masks instead of boxes or points.
            torch_model = predictor.sam_prompt_encoder.mask_downscaling
        case "not supported: memory-attention":
            torch_model = predictor.memory_attention
        case _:
            print(f"Unknown model block: {block}")
            exit()
    
    if len(inputs) == 1:
        torch_input = torch.randn(inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3])
    else:
        torch_input = list()
        for e in inputs:
            if len(e) == 4:
                torch_input.append(torch.randn(e[0], e[1], e[2], e[3]))
            elif len(e) == 3:
                torch_input.append(torch.randn(e[0], e[1], e[2]))
    
    if False:
        # Currently only works if models forward function only has a single input parameter
        test_torch_model(torch_model, torch_input)

    if additional_parameters is not None:
        for p in additional_parameters:
            torch_input.append(p)

    if separate_parameters:
        torch_input = tuple(torch_input)

    export_path = out_dir / Path(f"{m.name}-{block}-sa1-v01.onnx")
    torch.onnx.export(torch_model, torch_input, export_path, 
                      export_params=True,
                      opset_version=17, 
                      input_names=['input']
                      )
    
    if use_simplify:
        model = onnx.load(export_path)
        model, check = onnxsim.simplify(model)
        if check:
            simplify_path = modify_filename(export_path)
            onnx.save(model, simplify_path)
        else:
            print("Model simplification failed!")
    analyze_onnx_model(onnx.load(export_path))
    # test_onnx_model(export_path, torch_input)

if __name__ == "__main__":
    models = [
            ModelSource("nanosam2-resnet18", "results/sam2.1_hiera_s_resnet18/checkpoint.pth", "../sam2_configs/nanosam2.1_resnet18.yaml"),
            ModelSource("nanosam2-mobilenetV3", "results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth", "../sam2_configs/nanosam2.1_mobilenet_v3_large.yaml")
            ]
    out_dir = Path("model_exports")

    export_blocks = [
        "image-encoder",
        "image-encoder-trunk",
        "image-encoder-neck",
        "mask-decoder-transformer",
        "memory-encoder",
    ]
    for b in export_blocks:
        export_model_block(models[0], b, out_dir, use_simplify=False)
    print("done.")


import onnx
from pathlib import Path
from tools.misc import ModelSource
import torch
from nanosam2.sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import numpy as np
import onnxruntime as ort

def modify_filename(file_path) -> Path:
    # Check if the input is already a Path object
    if not isinstance(file_path, Path):
        # Create a Path object from the file path string
        path = Path(file_path)
    else:
        path = file_path

    # Create the new file name by adding "modified" before the extension
    new_file_name = path.with_name(f"{path.stem}_modified{path.suffix}")

    return new_file_name

def remove_duplicate_outputs(model_path:str, out_file:str=None) -> onnx.ModelProto:
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

    

    #model.graph.output.remove()  # Clear existing outputs
    #model.graph.output.extend(new_outputs)  # Add the unique outputs
    if out_file is not None:
        onnx.save(model, out_file)
        print(f"Modified model saved to {out_file}")
    return {"onnx_model":model, "out_file": out_file}


def test_onnx_model(model_path:Path, input:np.ndarray):
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input})

    print("ONNX model output shapes:")
    for i, output in enumerate(outputs):
        print(f"{i} | shape: {output.shape}")

def export_model(m:ModelSource, out_dir:Path, xy:int=512):
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(m.cfg, m.checkpoint, torch.device("cpu"))
    torch_model = predictor.image_encoder
    torch_input = torch.randn(1, 3, xy, xy)
    y = torch_model(torch_input)

    print("Original model output shapes:")
    oid = 0
    for k,v in y.items():
        if not isinstance(v, list):
            print(f"{oid} | {k} shape: {v.shape}")
            oid += 1
        else:
            print(f"{k}: is type of list")
            for e in v:
                print(f"{oid} | - {e.shape}")
                oid += 1
    
    export_path = out_dir / Path(f"{m.name}-image-encoder-{xy}-sa1-v01.onnx")
    torch.onnx.export(torch_model, torch_input, export_path, 
                      opset_version=17, 
                      input_names=['input']
                      )
    test_onnx_model(export_path, torch_input.numpy())
    modified_model = remove_duplicate_outputs(export_path, modify_filename(export_path))
    print("Modified ", end='')
    test_onnx_model(modified_model["out_file"], torch_input.numpy())


if __name__ == "__main__":
    models = [
            ModelSource("nanosam2-resnet18", "results/sam2.1_hiera_s_resnet18/checkpoint.pth", "../sam2_configs/nanosam2.1_resnet18.yaml"),
            ModelSource("nanosam2-mobilenetV3", "results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth", "../sam2_configs/nanosam2.1_mobilenet_v3_large.yaml")
            ]
    out_dir = Path("model_exports")
    export_model(models[0], out_dir)
    print("done.")


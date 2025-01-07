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
            if isinstance(v, list) or isinstance(v, tuple):
                determine_torch_output_shapes(v)
            else:
                print(f"{id} | {k} shape: {v.shape}")
                id += 1
    elif isinstance(y, list) or isinstance(y, tuple):
        for k,v in enumerate(y):
            determine_torch_output_shapes(v)
    else:
        print(f"out | - {y.shape}")

def test_torch_model(torch_model:torch.nn, input, silent=False, use_unpack_operator=False) -> bool:
    if use_unpack_operator: y = torch_model(*input)
    else: y = torch_model(input)
    if not silent:
        determine_torch_output_shapes(y)
    return True


def get_block_and_inputs(predictor:torch.nn, block:str, img_shape:list=[3,512,512]):
    use_unpack_operator = True
    match block:
        case "nanosam2":
            print("Hint: Full model export not supported.")
            torch_model = predictor
            torch_input = (torch.randn(1,img_shape[0],img_shape[1],img_shape[2]),)
        case "image-encoder":
            torch_model = predictor.image_encoder
            torch_input = (torch.randn(1,img_shape[0],img_shape[1],img_shape[2]),)
        case "image-encoder-trunk":
            torch_model = predictor.image_encoder.trunk
            torch_input = (torch.randn(1,img_shape[0],img_shape[1],img_shape[2]),)
        case "image-encoder-neck":
            torch_model = predictor.image_encoder.neck
            torch_input = [torch.randn(1,64,128,128),
                           torch.randn(1,128,64,64),
                           torch.randn(1,256,32,32),
                           torch.randn(1,512,16,16)]
            use_unpack_operator = False
        case "mask-decoder":
            torch_model = predictor.sam_mask_decoder
            torch_input = (torch.randn(1, 256, 32, 32),
                           torch.randn(1, 256, 32, 32),
                           torch.randn(1, 8, 256))
        case "mask-decoder-transformer":
            torch_model = predictor.sam_mask_decoder.transformer
            torch_input = (torch.randn(1, 256, 32, 32),
                           torch.randn(1, 256, 32, 32),
                           torch.randn(1, 8, 256))
        case "memory-encoder":
            torch_model = predictor.memory_encoder
            torch_input = (torch.randn(1, 256, 32, 32),
                           torch.randn(1, 1, 512, 512),
                           True)
        case "not supported: prompt-encoder-mask-downscaling":
            # mask_downscaling is only used in SAM2 when prompting with masks instead of boxes or points.
            torch_model = predictor.sam_prompt_encoder.mask_downscaling
        case "memory-attention":
            print("Hint: Fails due to implementation of memory_attention. Value p is for the number of prompts entered.")
            p = 2
            torch_model = predictor.memory_attention
            torch_input = ([torch.randn(1024, p, 256)],
                           torch.randn(7200, p, 64),
                           [torch.randn(1024, p, 256)],
                           torch.randn(7200, p, 64),
                           32)
        case _:
            print(f"Unknown model block: {block}")
            exit()
    return (torch_model, torch_input, use_unpack_operator)

def export_model_block(m:ModelSource, block:str, out_dir:Path, img_shape:list, use_simplify:bool=False):
    """
    Export a building block of nanosam2 to onnx. Not all blocks can be converted.
    """
    print(f"Exporting Model: {m.name} block: {block}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(m.cfg, m.checkpoint, torch.device("cpu"))
    torch_model, torch_input, use_unpack_operator = get_block_and_inputs(predictor=predictor, block=block, img_shape=img_shape)
      
    print(" - Testing torch model...", end="")
    test_torch_model(torch_model, torch_input, silent=True, use_unpack_operator=use_unpack_operator)
    print("OK")

    print(" - Exporting to ONNX...", end="")
    export_path = out_dir / Path(f"{m.name}-{block}-sa1-v01.onnx")
    torch.onnx.export(torch_model, torch_input, export_path, 
                      export_params=True,
                      opset_version=17, 
                      input_names=['input']
                      )
    print("OK")
    print(f" - Model stored in {export_path}")
    
    if use_simplify:
        print(" - simplify model...", end="")
        model = onnx.load(export_path)
        model, check = onnxsim.simplify(model)
        if check:
            export_path = modify_filename(export_path)
            onnx.save(model, export_path)
            print("OK")
            print(f" - Simplify model stored in {export_path}")
        else:
            print("Model simplification failed!")

    print(" - Analyze model...")
    analyze_onnx_model(onnx.load(export_path))
    print(" - Model block successfully converted to ONNX.")
    # test_onnx_model(export_path, torch_input)

if __name__ == "__main__":
    import argparse
    print("""
          Welcome to Nanosam2 ONNX exporter!
          Nanosam2 ONNX exporter is used to export parts of the Nanosam2 model to ONNX files.
          These ONNX files can be executed using onnxruntime or further processed and deployed on the desired hardware.

           - All shapes used in this ONNX exporter are determined with an input shape of 1,3,512,512
          """)
    models = [
            ModelSource("sam2.1_small", "results/sam2.1_hiera_s/sam2.1_hiera_small.pt", "../sam2_configs/sam2.1_hiera_s.yaml"),
            ModelSource("nanosam2-resnet18", "results/sam2.1_hiera_s_resnet18/checkpoint.pth", "../sam2_configs/nanosam2.1_resnet18.yaml"),
            ModelSource("nanosam2-mobilenetV3", "results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth", "../sam2_configs/nanosam2.1_mobilenet_v3_large.yaml")
            ]

    valid_exports = [
        "nanosam2",
        "all",
        "image-encoder",
        "image-encoder-trunk",
        "image-encoder-neck",
        "mask-decoder",
        "mask-decoder-transformer",
        "memory-encoder",
        "memory-attention"
    ]

    valid_encoders = [
        "hiera_small",
        "resnet18",
        "mobilenetV3_large",
        "casvit_s"
    ]

    parser = argparse.ArgumentParser("Nanosam2 ONNX exporter")
    parser.add_argument("--export", type=str, default="image-encoder", choices=valid_exports, help='Model block to export, use all to export all supported blocks as individuals, use nanosam2 to export the whole model.')
    parser.add_argument("--output_path", type=str, default="model_exports2", help="Export directory.")
    parser.add_argument("--img_shape", nargs='+', type=int, default=[3,512,512], help="Image shape to use.")
    parser.add_argument("--encoder_type", type=str, default="resnet18", choices=valid_encoders)

    out_dir = Path("model_exports2")

    args = parser.parse_args()

    match args.encoder_type:
        case "hiera_small":
            model = models[0]
        case "resnet18":
            model = models[1]
        case "mobilenetV3_large":
            model = models[2]
        case "casvit_s":
            model = models[3]
        case _:
            print("Enocer type not supported.")
            exit(1)

    if not Path(model.checkpoint).is_file():
        print(f"Path to model {model.checkpoint} not found.")
        exit(1)
    
    if args.export != "all":
        # Export a single block or the whole nanosam2 model.
        export_model_block(model, args.export, out_dir, args.img_shape, use_simplify=False)
    else:
        # Export all blocks as individuals.
        for b in valid_exports:
            if b == "all": continue
            export_model_block(model, b, out_dir, args.img_shape, use_simplify=False)
    print("done.")

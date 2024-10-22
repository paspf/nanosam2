import argparse
import torch
import time
import copy

from nanosam2.sam2.build_sam import build_sam2

def run_benchmark(encoder, setting):

    device, dtype, num_runs, bs = setting
    
    encoder = copy.deepcopy(encoder)
    input = torch.ones((bs, 3, sam21.image_size, sam21.image_size)).to(device)
    encoder.to(device)

    with torch.autocast(device, dtype=dtype):
        for _ in range(2):
            encoder(input.to(device))

        start = time.perf_counter()
        for _ in range(num_runs):
            fmaps = encoder(input)
        end = time.perf_counter()

    s_per_sample = (end-start) / num_runs / bs

    print(f"Average FPS: {1/s_per_sample:.1f} for dtype {dtype} on device {device}")

    return s_per_sample

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t", 
                        help="The path to a corresponding config.")
    args = parser.parse_args()

    sam21 = build_sam2(args.config, mode="eval")

    bench_settings = [["cpu", torch.float32, 50, 1], 
                      ["cuda", torch.float32, 100, 1],
                      ["cuda", torch.float16, 100, 1],
                      ["cuda", torch.bfloat16, 100, 1]]

    with torch.inference_mode(True):
        for setting in bench_settings:
            run_benchmark(sam21.image_encoder, setting)

    # compile model
    encoder_compiled = torch.compile(sam21.image_encoder)

    print("Testing torch.compile")
    with torch.inference_mode(True):
        for setting in bench_settings:
            run_benchmark(encoder_compiled, setting)
    



    





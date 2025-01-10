# Benchmark the video performance on different devices using different compiler optimizations.
#
# Pass a video, a list of devices to test, a list of flags, deciding which network block to optimize, the models
# and the number of iterations to BenchmarkVideoPerformance. Call the run() member function to run the benchmarks.
#
# Example:
# Run benchmarks on the following dataset:
# python nanosam2/tools/benchmark_video_performance.py "../../datasets/ImageSeries1"

import os
import numpy as np
import torch
import itertools
import time
from nanosam2.sam2.build_sam import build_sam2_video_predictor
from nanosam2.datasets.containers import ModelSource

class BenchmarkVideoPerformance:
    class BenchmarkIterationMetadata:
        def __init__(
                self,
                model:ModelSource, 
                device, 
                compile_settings:list,
                compile_backend:str
                ):
            self.model = model
            self.device = device
            self.compile_encoder = compile_settings[0]
            self.compile_memory_attention = compile_settings[1]
            self.compile_sam_mask_decoder = compile_settings[2]
            self.compile_sam_prompt_encoder = compile_settings[3]
            self.compile_memory_encoder = compile_settings[4]
            self.compile_backend = compile_backend

            # Results
            self.total_runtime = None
            self.time_per_ds = None
            self.s_per_frame = None

    def __init__(
        self,
        video_dir:str,
        devices:list,
        models:list,
        compile_settings:list=None,
        compile_backend:str="inductor",
        iterations:int=1):
        self.video_dir = video_dir
        self.devices = devices
        self.models = models
        self.compile_settings = compile_settings
        self.compile_backend = compile_backend
        self.iterations = iterations
    
        # Check if auto devices is active:
        if isinstance(self.devices[0], str):
            if self.devices[0] == "auto":
                self.devices = []
                self.devices.append(torch.device("cpu"))
                if torch.cuda.is_available():
                    self.devices.append(torch.device("cuda"))
                if torch.backends.mkldnn.is_available():
                    self.devices.append(torch.device("mkldnn"))
        print(self.devices)
        # Benchmark Statistics
        self.b_stats = []

    def hardware_prep(self, device):
        # Hardware Prep
        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def run(self):
        print("Preparing benchmarks...")

        self.frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if len(self.frame_names) == 0:
            print("Frame directory is empty!")
            exit()

        # Generate compilation options if not given.
        if self.compile_settings is None:
            self.compile_settings = self._generate_combinations(5)
        
        num_of_benchmarks = len(self.models) * len(self.devices) * len(self.compile_settings)
        
        b_counter = 1
        for m in self.models:
            for d in self.devices:
                self.hardware_prep(d)
                for cs in self.compile_settings:
                    self.b_stats.append(self.inference(self.BenchmarkIterationMetadata(m, d, cs, self.compile_backend)))
                    print(f"Running benchmark {b_counter}/{num_of_benchmarks} ")
                    b_counter += 1



    def inference(self, 
                  bim:BenchmarkIterationMetadata
                  ) -> BenchmarkIterationMetadata:
        predictor = build_sam2_video_predictor(bim.model.cfg, bim.model.checkpoint, device=bim.device)

        
        # Compile Model
        if bim.compile_encoder:
            predictor.image_encoder = torch.compile(predictor.image_encoder, backend=bim.compile_backend, dynamic=False)
        if bim.compile_memory_attention:
            predictor.memory_attention = torch.compile(predictor.memory_attention, backend=bim.compile_backend)
        if bim.compile_sam_mask_decoder:
            predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, backend=bim.compile_backend)
        if bim.compile_sam_prompt_encoder:
            predictor.sam_prompt_encoder = torch.compile(predictor.sam_prompt_encoder, backend=bim.compile_backend)
        if bim.compile_memory_encoder:
            predictor.memory_encoder = torch.compile(predictor.memory_encoder, backend=bim.compile_backend)

        inference_state = predictor.init_state(video_path=self.video_dir, disable_prints=True)
        predictor.reset_state(inference_state)

        # Select an Object
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 4  # give a unique id to each object

        # Let's add a positive click at (x, y) = (460, 60) to refine the mask
        points = np.array([[560, 350], [770, 420], [750, 380]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1,1,1], np.int32)
        # note that we also need to send the original box input along with
        # the new refinement click together into `add_new_points_or_box`
        box = np.array([400, 320, 1100, 650], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box,
        )

        # Perform dummy inference to activate model compilation
        video_segments = {}  # per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, disable_prints=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Run Benchmark
        start = time.perf_counter()
        for _ in range(self.iterations):
            # Run inferences
            video_segments = {}  # per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, disable_prints=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        end = time.perf_counter()

        bim.total_runtime = (end-start)
        bim.time_per_ds = bim.total_runtime / self.iterations
        bim.s_per_frame = bim.time_per_ds / len(self.frame_names)
        return bim

    def print_results(self):
        best_run_id = 0
        best_fps=0
        i = 0
        for bim in self.b_stats:
            print(f"| {i} | {bim.model.name} | {bim.device} | {bim.compile_encoder} | {bim.compile_memory_attention} | {bim.compile_sam_mask_decoder}"
              f" | {bim.compile_sam_prompt_encoder} | {bim.compile_memory_encoder} |", end='')
            fps = 1/bim.s_per_frame
            print(f" {round(bim.time_per_ds, 3)} | {round(fps, 3)} |")
            if fps > best_fps:
                best_fps = fps
                best_run_id = i
            i +=1
        print(f"best run: {best_run_id} with {best_fps} fps.")

    def _generate_combinations(self, n):
        # Compute the cartesian product of input iterables.
        return [list(x) for x in itertools.product([False, True], repeat=n)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmarks on different devices with different model optimizations.')
    parser.add_argument('video', help='Path of the video frames.')
    args = parser.parse_args()

    if args.video is None:
            print("Please provide video path.")
            exit()

    use_zentorch = False
    if use_zentorch:
        # https://www.amd.com/en/developer/zendnn.html
        import zentorch
        compile_backend="zentorch"
    else:
        compile_backend="inductor"

    
    devices = [torch.device("cuda")]
    # devices = ["auto"]

    models = [
        ModelSource("nanosam2_resnet18", "results/sam2.1_hiera_s_resnet18/checkpoint.pth", "../sam2_configs/nanosam2.1_resnet18.yaml"),
        ModelSource("sam2.1_small", "results/sam2.1_hiera_s/sam2.1_hiera_small.pt", "../sam2_configs/sam2.1_hiera_s.yaml"),
        ModelSource("nanosam2_mobilenetV3", "results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth", "../sam2_configs/nanosam2.1_mobilenet_v3_large.yaml")
        ]
    bvp = BenchmarkVideoPerformance(args.video, devices, models, compile_settings=[[False, False, False, False, False],
                                                                                   [True, False, False, False, False],
                                                                                   [True, True, False, False, True]], 
                                                                                   compile_backend=compile_backend,
                                                                                   iterations=5)
    #bvp = BenchmarkVideoPerformance(args.video, devices, models, iterations=5)
    bvp.run()
    bvp.print_results()

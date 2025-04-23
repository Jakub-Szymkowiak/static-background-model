import os
import shutil
import subprocess

from argparse import ArgumentParser
from abc import ABC, abstractmethod


class MediaHandler:
    def __init__(self, src_path: str, out_path: str):
        self.source = src_path
        self.output = out_path
        self.out_source = os.path.join(self.output, "source")
        self.input_type = self._detect_input_type()

    def _detect_input_type(self) -> str:
        video_dir = os.path.join(self.source, "video")
        image_dir = os.path.join(self.source, "images")

        has_mp4 = all(f.endswith(".mp4") for f in os.listdir(video_dir)) if os.path.isdir(video_dir) else False
        has_jpg = all(f.endswith(".jpg") for f in os.listdir(image_dir)) if os.path.isdir(image_dir) else False

        if has_mp4:
            # first .mp4 file found, please provide a single .mp4 file as input
            self.source = os.path.join(video_dir, next(f for f in os.listdir(video_dir) if f.endswith(".mp4")))
            return "mp4"
        elif has_jpg:
            self.source = image_dir
            return "jpg"

        raise ValueError("Expected either video/*.mp4 or images/*.jpg inside the source directory.")


    def copy_data(self):
        os.makedirs(self.out_source, exist_ok=True)
        if self.input_type == "jpg":
            img_dir = os.path.join(self.out_source, "images")
            os.makedirs(img_dir, exist_ok=True)
            for file in os.listdir(self.source):
                src_file = os.path.join(self.source, file)
                dst_file = os.path.join(img_dir, file)
                shutil.copy2(src_file, dst_file)
        else:
            video_dir = os.path.join(self.out_source, "video")
            os.makedirs(video_dir, exist_ok=True)
            dst_file = os.path.join(video_dir, "video.mp4")
            shutil.copy2(self.source, dst_file)

    def convert_media(self, fps: int=24):
        callbacks = {
            "mp4": lambda: self.mp4_to_jpg(self.source, self.output),
            "jpg": lambda: self.jpg_to_mp4(self.source, os.path.join(self.output, "video.mp4"), fps)
        }
        if self.input_type not in callbacks:
            raise ValueError(f"Unsupported input type: {self.input_type}")
        print(f"Converting {self.input_type} using FFmpeg...")
        callbacks[self.input_type]()

    @staticmethod
    def mp4_to_jpg(input_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        output_pattern = os.path.join(output_dir, "frame_%05d.jpg")
        command = [
            "/usr/bin/ffmpeg",
            "-i", input_path,
            "-qscale:v", "2",
            output_pattern
        ]
        run_command(command)

    @staticmethod
    def jpg_to_mp4(input_dir: str, output_path: str, fps: int=24):
        input_pattern = os.path.join(input_dir, "%05d.jpg")
        command = [
            "/usr/bin/ffmpeg",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        run_command(command)

class InpaintingHandler(ABC):
    def __init__(self, src_path, out_path):
        self.src_path = src_path
        self.out_path = out_path
        self.mask_path = os.path.join(src_path, "masks")
        self.mp4_path = os.path.join(out_path, "video.mp4")
        self.jpg_path = os.path.join(out_path, "images")

    @abstractmethod
    def run(self):
        pass


class E2FGVIHandler(InpaintingHandler):
    model_name = "E2FGVI"
    model_variant = "e2fgvi_hq"
    ckpt_path = "release_model/E2FGVI-HQ-CVPR22.pth"
    working_dir = "/home/computergraphics/Documents/jszymkowiak/mono-vid-dynamic-gs/code/static-background-model/E2FGVI"
    out_dir = "/home/computergraphics/Documents/jszymkowiak/mono-vid-dynamic-gs/code/static-background-model/E2FGVI/results"

    def __init__(self, src_path, out_path):
        super().__init__(src_path, out_path)

    def run(self):
        print("Running inpainting pipeline.")
        print(f"Inpainting model: {self.model_name}.")

        print(self.mp4_path)

        command = [
            "python", "test.py",
            "--model", self.model_variant,
            "--video", self.mp4_path,
            "--mask", self.mask_path,
            "--ckpt", self.ckpt_path
        ]
        run_command(command, cwd=self.working_dir)

        self.output_handler()

    def output_handler(self):
        out_path = os.path.join(self.out_dir, "video_results.mp4")
        dest_path = os.path.join(self.out_path, "inpainted.mp4")

        # postprocessing
        command = [
            "/usr/bin/ffmpeg",
            "-i", out_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            dest_path
        ]
        run_command(command)
        
def run_command(command, **kwargs):
    cwd = kwargs.get("cwd", None)
    result = subprocess.run(command, cwd=cwd)

    # if result.returncode != 0:
    #     raise RuntimeError(f"Failed: {result}")
   
def parse_args():
    parser = ArgumentParser(description="Preprocessing pipeline.")

    parser.add_argument("-s", type=str, required=True, help="Path to source directory or video file")
    parser.add_argument("--model", type=str, default="E2FGVI", help="Inpainting model (E2FGVI or LaMa)")

    args, _ = parser.parse_known_args()
    scene_name = os.path.basename(os.path.normpath(args.s))
    default_output = f"./output/{scene_name}"

    parser.add_argument("-o", type=str, default=default_output, help="Path to output directory")

    return parser.parse_args()

def run_preprocessing_pipeline():
    MODEL_HANDLERS = {"E2FGVI": E2FGVIHandler}

    args = parse_args()

    media_handler = MediaHandler(args.s, args.o)
    media_handler.copy_data()
    media_handler.convert_media() 

    inpainting_handler = MODEL_HANDLERS[args.model](args.s, args.o)
    inpainting_handler.run()



if __name__ == "__main__":
    run_preprocessing_pipeline()

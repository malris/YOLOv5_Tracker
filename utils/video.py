from typing import Optional

import cv2
import numpy as np
import os


class Video:
    def __init__(self, input_path: str, output_path: Optional[str], codec_fourcc: Optional[str] = None, output_fps: int = 24,
                 output_video=None):
        self.input_path = input_path
        self.video_capture = cv2.VideoCapture(input_path)

        self.output_video = output_video
        self.output_path = output_path if output_path is not None else '.'

        self.codec_fourcc = codec_fourcc
        self.output_fps = output_fps

    def __iter__(self):
        while True:
            ret, frame = self.video_capture.read()
            if ret is False or frame is None:
                break

            yield frame

        if self.output_video is not None:
            self.output_video.release()
            print(
                f"Output video file saved to: {self.get_output_file_path()}"
            )
        self.video_capture.release()
        cv2.destroyAllWindows()

    def write(self, frame: np.array):
        if self.output_video is None:
            output_file_path = self.get_output_file_path()
            fourcc = cv2.VideoWriter_fourcc(*self.get_codec_fourcc(output_file_path))

            output_size = (frame.shape[1], frame.shape[0])
            self.output_video = cv2.VideoWriter(
                output_file_path,
                fourcc,
                self.output_fps,
                output_size,
            )

        self.output_video.write(frame)

    def get_output_file_path(self) -> str:
        if not os.path.isdir(self.output_path):
            return self.output_path

        if self.input_path is not None:
            base_file_name = self.input_path.split("/")[-1].split(".")[0]
            file_name = base_file_name + "_out.mp4"
        else:
            file_name = "out.mp4"

        return os.path.join(self.output_path, file_name)

    def get_codec_fourcc(self, filename: str) -> Optional[str]:
        if self.codec_fourcc is not None:
            return self.codec_fourcc

        extension = filename[-3:].lower()
        if "avi" == extension:
            return "XVID"
        elif "mp4" == extension:
            return "mp4v"
        else:
            print(
                f"Could not determine video codec for the provided output filename: "
                f"{filename}\n"
                f"Please use '.mp4', '.avi', or provide a custom OpenCV fourcc codec name."
            )
            return None

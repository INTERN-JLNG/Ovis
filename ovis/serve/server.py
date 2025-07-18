import argparse
import os.path
import gradio as gr
import numpy as np
import PIL

from gradio.components import Textbox, Image, Video

from ovis.serve.runner import RunnerArguments, OvisRunner
from moviepy.editor import VideoFileClip

class Server:
    def __init__(self, runner: OvisRunner):
        self.runner = runner

    def __call__(self, type, image, video, text):
        if type == "Image":
            inputs = [image, text]
        elif type == "Video":
            def _sampling_idx(_len, _min, _max):
                if _len < _min or _len > _max:
                    tgt_len = _min if _len < _min else _max
                    stride = _len / tgt_len
                    sampled_ids = []
                    for i in range(tgt_len):
                        start = int(np.round(stride * i))
                        end = int(np.round(stride * (i + 1)))
                        sampled_ids.append(min(_len - 1, (start + end) // 2))
                    return sampled_ids
                else:
                    return list(range(_len))

            with VideoFileClip(video) as clip:
                total_frames = int(clip.fps * clip.duration)
                sampled_ids = _sampling_idx(total_frames, 16, 16)
                frames = [clip.get_frame(idx / clip.fps) for idx in sampled_ids]
                frames = [PIL.Image.fromarray(frame, mode='RGB') for frame in frames]
                inputs = frames + [text]
        elif type == "TextOnly":
            inputs = [text]
        response = self.runner.run(inputs)
        output = response["output"]
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ovis Server')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--flagging_dir', type=str, default=os.path.expanduser('~/ovis-flagged'))
    parser.add_argument('--max_partition', type=int, default=9)
    parser.add_argument('--port', type=int, required=True)
    
    # ECP Two-Stage Reasoning Arguments
    parser.add_argument('--enable_ecp_two_stage', action='store_true', 
                       help='Enable ECP two-stage reasoning')
    parser.add_argument('--ecp_perception_max_new_tokens', type=int, default=256,
                       help='Max tokens for perception stage')
    parser.add_argument('--ecp_perception_temperature', type=float, default=0.7,
                       help='Temperature for perception stage')
    
    args = parser.parse_args()

    os.makedirs(args.flagging_dir, exist_ok=True)
    runner_args = RunnerArguments(
        model_path=args.model_path,
        max_partition=args.max_partition,
        enable_ecp_two_stage=args.enable_ecp_two_stage,
        ecp_perception_max_new_tokens=args.ecp_perception_max_new_tokens,
        ecp_perception_temperature=args.ecp_perception_temperature
    )
    
    # Create interface title with ECP indicator
    title = args.model_path.split('/')[-1]
    if args.enable_ecp_two_stage:
        title += " (ECP Two-Stage Reasoning)"
    
    demo = gr.Interface(
        fn=Server(OvisRunner(runner_args)),
        inputs=[gr.Radio(["Image", "Video", "TextOnly"], label="Choose Modality Type"),
                Image(type='pil', label='image'),
                Video(label='video', format='mp4'),
                Textbox(placeholder='Enter your text here...', label='prompt')],
        outputs=gr.Markdown(),
        title=title,
        flagging_dir=args.flagging_dir
    )
    demo.launch(server_port=args.port)

from dataclasses import field, dataclass
from typing import Optional, Union, List

import torch
from PIL import Image

from ovis.model.modeling_ovis import Ovis
from ovis.util.constants import IMAGE_TOKEN, ECP_PERCEPTION_PROMPT_DEFAULT, ECP_REASONING_PROMPT_TEMPLATE


@dataclass
class RunnerArguments:
    model_path: str
    max_new_tokens: int = field(default=512)
    do_sample: bool = field(default=False)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    temperature: Optional[float] = field(default=None)
    max_partition: int = field(default=9)
    
    # ECP Two-Stage Reasoning Parameters
    enable_ecp_two_stage: bool = field(default=False)
    ecp_perception_max_new_tokens: int = field(default=256)
    ecp_perception_temperature: Optional[float] = field(default=0.7)
    ecp_perception_prompt: Optional[str] = field(default=None)


class OvisRunner:
    def __init__(self, args: RunnerArguments):
        self.model_path = args.model_path
        self.dtype = torch.bfloat16
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = Ovis.from_pretrained(self.model_path, torch_dtype=self.dtype, multimodal_max_length=32768)
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()
        self.image_placeholder = IMAGE_TOKEN
        self.max_partition = args.max_partition
        self.gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        
        # ECP Two-Stage Reasoning Configuration
        self.enable_ecp_two_stage = args.enable_ecp_two_stage
        self.ecp_perception_prompt = args.ecp_perception_prompt or ECP_PERCEPTION_PROMPT_DEFAULT
        self.ecp_perception_gen_kwargs = dict(
            max_new_tokens=args.ecp_perception_max_new_tokens,
            do_sample=True,
            temperature=args.ecp_perception_temperature,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )

    def preprocess(self, inputs: List[Union[Image.Image, str]]):
        # for single image and single text inputs, ensure image ahead
        if len(inputs) == 2 and isinstance(inputs[0], str) and isinstance(inputs[1], Image.Image):
            inputs = reversed(inputs)

        # build query
        query = ''
        images = []
        for data in inputs:
            if isinstance(data, Image.Image):
                query += self.image_placeholder + '\n'
                images.append(data)
            elif isinstance(data, str):
                query += data.replace(self.image_placeholder, '')
            elif data is not None:
                raise RuntimeError(f'Invalid input type, expected `PIL.Image.Image` or `str`, but got {type(data)}')

        # format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=self.max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        if pixel_values is not None:
            pixel_values = [pixel_values.to(device=self.device, dtype=self.dtype)]
        else:
            pixel_values = [None]

        return prompt, input_ids, attention_mask, pixel_values

    def run(self, inputs: List[Union[Image.Image, str]]):
        """
        Main run method that can operate in single-stage or ECP two-stage mode.
        """
        if self.enable_ecp_two_stage:
            return self.run_ecp_two_stage(inputs)
        else:
            return self.run_single_stage(inputs)
    
    def run_single_stage(self, inputs: List[Union[Image.Image, str]]):
        """
        Standard single-stage processing (original Ovis behavior).
        """
        prompt, input_ids, attention_mask, pixel_values = self.preprocess(inputs)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.gen_kwargs
            )
        output = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        input_token_len = input_ids.shape[1]
        output_token_len = output_ids.shape[1]
        response = dict(
            prompt=prompt,
            output=output,
            prompt_tokens=input_token_len,
            total_tokens=input_token_len + output_token_len
        )
        return response
    
    def run_ecp_two_stage(self, inputs: List[Union[Image.Image, str]]):
        """
        ECP two-stage reasoning: perception stage followed by reasoning stage.
        """
        # Extract images and text from inputs
        images = [item for item in inputs if isinstance(item, Image.Image)]
        texts = [item for item in inputs if isinstance(item, str)]
        
        if not images:
            # No images, fall back to standard processing
            return self.run_single_stage(inputs)
        
        if not texts:
            # Only images, use default query
            original_query = "What do you see in this image?"
        else:
            # Use the first text as the main query
            original_query = texts[0]
        
        # Stage 1: Perception
        perception_result = self.run_perception_stage(images)
        
        # Stage 2: Reasoning
        final_response = self.run_reasoning_stage(perception_result, original_query, images)
        
        return final_response
    
    def run_perception_stage(self, images: List[Image.Image]) -> str:
        """
        Run the perception stage to generate visual understanding.
        """
        # Prepare perception stage inputs
        perception_inputs = []
        for image in images:
            perception_inputs.append(image)
        perception_inputs.append(self.ecp_perception_prompt)
        
        # Run perception stage
        prompt, input_ids, attention_mask, pixel_values = self.preprocess(perception_inputs)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.ecp_perception_gen_kwargs
            )
        
        # Decode perception output
        perception_output = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        prompt_length = len(prompt)
        perception_result = perception_output[prompt_length:].strip()
        
        return perception_result
    
    def run_reasoning_stage(self, perception_result: str, original_query: str, images: List[Image.Image]) -> dict:
        """
        Run the reasoning stage using the perception result and original question.
        """
        # Construct reasoning stage prompt
        reasoning_prompt = ECP_REASONING_PROMPT_TEMPLATE.format(
            perception_result=perception_result,
            original_query=original_query
        )
        
        # Prepare reasoning stage inputs (include images for context)
        reasoning_inputs = []
        for image in images:
            reasoning_inputs.append(image)
        reasoning_inputs.append(reasoning_prompt)
        
        # Run reasoning stage
        prompt, input_ids, attention_mask, pixel_values = self.preprocess(reasoning_inputs)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.gen_kwargs
            )
        
        # Decode and prepare final response
        reasoning_output = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        input_token_len = input_ids.shape[1]
        output_token_len = output_ids.shape[1]
        
        response = dict(
            prompt=prompt,
            output=reasoning_output,
            prompt_tokens=input_token_len,
            total_tokens=input_token_len + output_token_len,
            perception_result=perception_result,
            reasoning_prompt=reasoning_prompt,
            original_query=original_query,
            ecp_two_stage=True
        )
        
        return response


if __name__ == '__main__':
    runner_args = RunnerArguments(model_path='<model_path>')
    runner = OvisRunner(runner_args)
    image = Image.open('<image_path>')
    text = '<prompt>'
    response = runner.run([image, text])
    print(response['output'])

"""
ECP (Explicit Chaining of Perception) Two-Stage Reasoning Runner

This module implements the ECP approach for visual reasoning, which performs
inference in two stages:
1. Perception Stage: Generate detailed visual descriptions and understanding
2. Reasoning Stage: Use the visual understanding to answer the original question
"""

from dataclasses import field, dataclass
from typing import Optional, Union, List
import torch
from PIL import Image

from ovis.serve.runner import OvisRunner, RunnerArguments
from ovis.util.constants import IMAGE_TOKEN


@dataclass
class ECPRunnerArguments(RunnerArguments):
    """Arguments for ECP Two-Stage Reasoning Runner"""
    # Perception stage parameters
    perception_max_new_tokens: int = field(default=256)
    perception_temperature: Optional[float] = field(default=0.7)
    
    # Two-stage mode flag
    enable_two_stage: bool = field(default=True)
    
    # Custom perception prompts
    perception_prompt_template: str = field(
        default="Please provide a detailed description of this image, including objects, actions, relationships, and any text visible. Focus on visual elements that would be important for understanding and reasoning about the image."
    )


class ECPRunner(OvisRunner):
    """
    ECP (Explicit Chaining of Perception) Runner that implements two-stage visual reasoning.
    
    The two-stage process:
    1. Perception Stage: Generate detailed visual understanding of the image
    2. Reasoning Stage: Use the visual understanding to answer the original question
    """
    
    def __init__(self, args: ECPRunnerArguments):
        super().__init__(args)
        self.enable_two_stage = args.enable_two_stage
        self.perception_prompt_template = args.perception_prompt_template
        
        # Perception stage generation parameters
        self.perception_gen_kwargs = dict(
            max_new_tokens=args.perception_max_new_tokens,
            do_sample=True,
            temperature=args.perception_temperature,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
    
    def run_perception_stage(self, images: List[Image.Image]) -> str:
        """
        Run the perception stage to generate visual understanding.
        
        Args:
            images: List of PIL Images to analyze
            
        Returns:
            Generated visual description and understanding
        """
        # Prepare perception stage inputs
        perception_inputs = []
        for image in images:
            perception_inputs.append(image)
        perception_inputs.append(self.perception_prompt_template)
        
        # Run perception stage
        prompt, input_ids, attention_mask, pixel_values = self.preprocess(perception_inputs)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.perception_gen_kwargs
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
        
        Args:
            perception_result: Visual understanding from perception stage
            original_query: Original user question
            images: Original images (for reference)
            
        Returns:
            Final reasoning result
        """
        # Construct reasoning stage prompt
        reasoning_prompt = f"""Based on the following visual description:

{perception_result}

Please answer this question: {original_query}

Use the visual description to provide a detailed and accurate answer."""
        
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
            reasoning_prompt=reasoning_prompt
        )
        
        return response
    
    def run(self, inputs: List[Union[Image.Image, str]]) -> dict:
        """
        Main run method that can operate in single-stage or two-stage mode.
        
        Args:
            inputs: List of images and text inputs
            
        Returns:
            Response dict with generation results
        """
        if not self.enable_two_stage:
            # Fall back to standard single-stage processing
            return super().run(inputs)
        
        # Extract images and text from inputs
        images = [item for item in inputs if isinstance(item, Image.Image)]
        texts = [item for item in inputs if isinstance(item, str)]
        
        if not images:
            # No images, fall back to standard processing
            return super().run(inputs)
        
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
    
    def run_single_stage(self, inputs: List[Union[Image.Image, str]]) -> dict:
        """
        Explicitly run in single-stage mode (standard Ovis behavior).
        
        Args:
            inputs: List of images and text inputs
            
        Returns:
            Standard Ovis response
        """
        return super().run(inputs)


# Convenience function for creating ECP runner
def create_ecp_runner(model_path: str, **kwargs) -> ECPRunner:
    """
    Create an ECP runner with default or custom arguments.
    
    Args:
        model_path: Path to the Ovis model
        **kwargs: Additional arguments for ECPRunnerArguments
        
    Returns:
        Configured ECPRunner instance
    """
    args = ECPRunnerArguments(model_path=model_path, **kwargs)
    return ECPRunner(args)


if __name__ == '__main__':
    # Example usage
    from PIL import Image
    
    # Create ECP runner
    runner = create_ecp_runner(
        model_path='AIDC-AI/Ovis2-8B',  # Replace with actual model path
        enable_two_stage=True,
        perception_max_new_tokens=256,
        max_new_tokens=512
    )
    
    # Example with image and question
    # image = Image.open('path/to/image.jpg')
    # response = runner.run([image, "What is happening in this image?"])
    # print("Perception:", response.get('perception_result'))
    # print("Final Answer:", response.get('output'))
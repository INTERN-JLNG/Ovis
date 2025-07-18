# ECP Two-Stage Visual Reasoning

This repository now includes ECP (Explicit Chaining of Perception) two-stage visual reasoning capabilities, inspired by Qwen2VL's approach.

## Quick Start

### Enable ECP in Standard Runner

```python
from PIL import Image
from ovis.serve.runner import RunnerArguments, OvisRunner

# Enable ECP two-stage reasoning
args = RunnerArguments(
    model_path='AIDC-AI/Ovis2-8B',
    enable_ecp_two_stage=True,
    ecp_perception_max_new_tokens=256
)

runner = OvisRunner(args)
image = Image.open('your_image.jpg')
response = runner.run([image, "What's happening in this image?"])

# Access both stages
print("Perception:", response['perception_result'])
print("Final Answer:", response['output'])
```

### Use Dedicated ECP Runner

```python
from ovis.serve.ecp_runner import create_ecp_runner

runner = create_ecp_runner(
    model_path='AIDC-AI/Ovis2-8B',
    enable_two_stage=True
)

response = runner.run([image, "Analyze this scene in detail"])
```

### Run ECP-Enhanced Server

```bash
python ovis/serve/server.py \
    --model_path AIDC-AI/Ovis2-8B \
    --port 7860 \
    --enable_ecp_two_stage \
    --ecp_perception_max_new_tokens 256
```

## Features

- **Two-Stage Reasoning**: Perception → Reasoning for better visual understanding
- **Backward Compatible**: All existing code continues to work
- **Customizable**: Domain-specific perception prompts supported
- **Transparent**: Access to intermediate perception results
- **Flexible**: Multiple usage patterns available

## Documentation

- [Complete ECP Documentation](docs/ecp_two_stage_reasoning.md)
- [Usage Examples](examples/ecp_usage_examples.py)
- [Implementation Details](ovis/serve/ecp_runner.py)

## Benefits

- Improved accuracy on complex visual reasoning tasks
- Better handling of multi-step visual analysis
- Transparent reasoning process
- Enhanced performance on domain-specific tasks (medical, scientific, etc.)

The ECP implementation is production-ready and maintains full compatibility with existing Ovis workflows.
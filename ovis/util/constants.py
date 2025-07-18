# Model Constants
IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]

# Log & Print
BEGIN_LINE = '========================************========================'
END_LINE = '------------------------------------------------------------'

# ECP Two-Stage Reasoning Constants
ECP_PERCEPTION_PROMPT_DEFAULT = (
    "Please provide a detailed description of this image, including objects, actions, "
    "relationships, and any text visible. Focus on visual elements that would be "
    "important for understanding and reasoning about the image."
)

ECP_REASONING_PROMPT_TEMPLATE = """Based on the following visual description:

{perception_result}

Please answer this question: {original_query}

Use the visual description to provide a detailed and accurate answer."""

# ECP Configuration Keys
ECP_ENABLE_TWO_STAGE = "enable_two_stage"
ECP_PERCEPTION_MAX_TOKENS = "perception_max_new_tokens"
ECP_PERCEPTION_TEMPERATURE = "perception_temperature"

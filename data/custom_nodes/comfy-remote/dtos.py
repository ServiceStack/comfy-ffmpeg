""" Options:
Date: 2025-06-17 18:07:24
Version: 8.81
Tip: To override a DTO option, remove "#" prefix before updating
BaseUrl: https://comfy-gateway.pvq.app

#GlobalNamespace: 
#AddServiceStackTypes: True
#AddResponseStatus: False
#AddImplicitVersion: 
#AddDescriptionAsComments: True
IncludeTypes: {Agent}
#ExcludeTypes: 
#DefaultImports: datetime,decimal,marshmallow.fields:*,servicestack:*,typing:*,dataclasses:dataclass/field,dataclasses_json:dataclass_json/LetterCase/Undefined/config,enum:Enum/IntEnum
#DataClass: 
#DataClassJson: 
"""

import datetime
import decimal
from marshmallow.fields import *
from servicestack import *
from typing import *
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, LetterCase, Undefined, config
from enum import Enum, IntEnum


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GpuInfo:
    index: int = 0
    name: Optional[str] = None
    total: int = 0
    free: int = 0
    used: int = 0


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class OllamaGenerateResponse:
    # @ApiMember(Description="The model used")
    model: Optional[str] = None
    """
    The model used
    """


    # @ApiMember(Description="The Unix timestamp (in seconds) of when the chat completion was created.")
    created_at: int = 0
    """
    The Unix timestamp (in seconds) of when the chat completion was created.
    """


    # @ApiMember(Description="The full response")
    response: Optional[str] = None
    """
    The full response
    """


    # @ApiMember(Description="Whether the response is done")
    done: bool = False
    """
    Whether the response is done
    """


    # @ApiMember(Description="The reason the response completed")
    done_reason: Optional[str] = None
    """
    The reason the response completed
    """


    # @ApiMember(Description="Time spent generating the response")
    total_duration: int = 0
    """
    Time spent generating the response
    """


    # @ApiMember(Description="Time spent in nanoseconds loading the model")
    load_duration: int = 0
    """
    Time spent in nanoseconds loading the model
    """


    # @ApiMember(Description="Time spent in nanoseconds evaluating the prompt")
    prompt_eval_count: int = 0
    """
    Time spent in nanoseconds evaluating the prompt
    """


    # @ApiMember(Description="Time spent in nanoseconds evaluating the prompt")
    prompt_eval_duration: int = 0
    """
    Time spent in nanoseconds evaluating the prompt
    """


    # @ApiMember(Description="Number of tokens in the response")
    eval_count: int = 0
    """
    Number of tokens in the response
    """


    # @ApiMember(Description="Time in nanoseconds spent generating the response")
    prompt_tokens: int = 0
    """
    Time in nanoseconds spent generating the response
    """


    # @ApiMember(Description="An encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory")
    context: Optional[List[int]] = None
    """
    An encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory
    """


    response_status: Optional[ResponseStatus] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class AgentEvent:
    name: Optional[str] = None
    args: Optional[Dict[str, str]] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class PromptInfo:
    client_id: Optional[str] = None
    prompt_id: Optional[str] = None
    api_prompt_url: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class OllamaGenerateOptions:
    # @ApiMember(Description="Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)")
    mirostat: Optional[int] = None
    """
    Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    """


    # @ApiMember(Description="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)")
    mirostat_eta: Optional[float] = None
    """
    Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)
    """


    # @ApiMember(Description="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)")
    mirostat_tau: Optional[float] = None
    """
    Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)
    """


    # @ApiMember(Description="Sets the size of the context window used to generate the next token. (Default: 2048)")
    num_ctx: Optional[int] = None
    """
    Sets the size of the context window used to generate the next token. (Default: 2048)
    """


    # @ApiMember(Description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)")
    repeat_last_n: Optional[int] = None
    """
    Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
    """


    # @ApiMember(Description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)")
    repeat_penalty: Optional[float] = None
    """
    Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
    """


    # @ApiMember(Description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")
    temperature: Optional[float] = None
    """
    The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
    """


    # @ApiMember(Description="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)")
    seed: Optional[int] = None
    """
    Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
    """


    # @ApiMember(Description="Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.\t")
    stop: Optional[str] = None
    """
    Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.	
    """


    # @ApiMember(Description="Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)")
    num_predict: Optional[int] = None
    """
    Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)
    """


    # @ApiMember(Description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
    top_k: Optional[int] = None
    """
    Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    """


    # @ApiMember(Description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
    top_p: Optional[float] = None
    """
    Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    """


    # @ApiMember(Description="Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)")
    min_p: Optional[float] = None
    """
    Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)
    """


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class ComfyTask:
    id: int = 0
    name: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetComfyAgentEventsResponse:
    results: List[AgentEvent] = field(default_factory=list)
    response_status: Optional[ResponseStatus] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class RegisterComfyAgentResponse:
    id: int = 0
    api_key: Optional[str] = None
    device_id: Optional[str] = None
    nodes: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    pending_prompts: List[PromptInfo] = field(default_factory=list)
    response_status: Optional[ResponseStatus] = None


# @Api(Description="Generate a response for a given prompt with a provided model.")
@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class OllamaGenerate:
    """
    Generate a response for a given prompt with a provided model.
    """

    # @ApiMember(Description="ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API")
    model: Optional[str] = None
    """
    ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API
    """


    # @ApiMember(Description="The prompt to generate a response for")
    prompt: Optional[str] = None
    """
    The prompt to generate a response for
    """


    # @ApiMember(Description="The text after the model response")
    suffix: Optional[str] = None
    """
    The text after the model response
    """


    # @ApiMember(Description="List of base64 images referenced in this request")
    images: Optional[List[str]] = None
    """
    List of base64 images referenced in this request
    """


    # @ApiMember(Description="The format to return a response in. Format can be `json` or a JSON schema")
    format: Optional[str] = None
    """
    The format to return a response in. Format can be `json` or a JSON schema
    """


    # @ApiMember(Description="Additional model parameters")
    options: Optional[OllamaGenerateOptions] = None
    """
    Additional model parameters
    """


    # @ApiMember(Description="System message")
    system: Optional[str] = None
    """
    System message
    """


    # @ApiMember(Description="The prompt template to use")
    template: Optional[str] = None
    """
    The prompt template to use
    """


    # @ApiMember(Description="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a `data: [DONE]` message")
    stream: Optional[bool] = None
    """
    If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a `data: [DONE]` message
    """


    # @ApiMember(Description="If `true` no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API")
    raw: Optional[bool] = None
    """
    If `true` no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API
    """


    # @ApiMember(Description="Controls how long the model will stay loaded into memory following the request (default: 5m)")
    keep_alive: Optional[str] = None
    """
    Controls how long the model will stay loaded into memory following the request (default: 5m)
    """


    # @ApiMember(Description="The context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory")
    context: Optional[List[int]] = None
    """
    The context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory
    """


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class ComfyTasksResponse:
    results: List[ComfyTask] = field(default_factory=list)
    response_status: Optional[ResponseStatus] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class UpdateComfyAgent(IReturn[EmptyResponse], IPost):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None

    queue_count: int = 0
    gpus: Optional[List[GpuInfo]] = None
    running_generation_ids: Optional[List[str]] = None
    queued_generation_ids: Optional[List[str]] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetComfyAgentEvents(IReturn[GetComfyAgentEventsResponse], IGet):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class RegisterComfyAgent(IReturn[RegisterComfyAgentResponse], IPost):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None

    version: int = 0
    workflows: List[str] = field(default_factory=list)
    queue_count: int = 0
    gpus: Optional[List[GpuInfo]] = None
    language_models: Optional[List[str]] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class UnRegisterComfyAgent(IReturn[EmptyResponse], IPost):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class UpdateWorkflowGeneration(IReturn[EmptyResponse], IPost):
    # @Validate(Validator="NotEmpty")
    id: Optional[str] = None

    # @Validate(Validator="NotEmpty")
    device_id: Optional[str] = None

    prompt_id: Optional[str] = None
    status: Optional[str] = None
    outputs: Optional[str] = None
    queue_count: Optional[int] = None
    error: Optional[ResponseStatus] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class CaptionArtifact(IReturn[EmptyResponse], IPost):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None

    # @Validate(Validator="NotEmpty")
    artifact_url: Optional[str] = None

    caption: Optional[str] = None
    description: Optional[str] = None


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetOllamaGenerateTask(IReturn[OllamaGenerate], IGet):
    # @Validate(Validator="GreaterThan(0)")
    id: int = 0


@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class CompleteOllamaGenerateTask(OllamaGenerateResponse, IReturn[EmptyResponse], IPost):
    # @Validate(Validator="GreaterThan(0)")
    id: int = 0


# @Route("/comfy/tasks")
@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetComfyTasks(IReturn[ComfyTasksResponse], IGet):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None


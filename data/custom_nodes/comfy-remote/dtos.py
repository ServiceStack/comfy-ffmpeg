""" Options:
Date: 2025-06-08 21:17:28
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
    tags: List[str] = field(default_factory=list)
    pending_prompts: List[PromptInfo] = field(default_factory=list)
    response_status: Optional[ResponseStatus] = None


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

    workflows: List[str] = field(default_factory=list)
    queue_count: int = 0
    gpus: Optional[List[GpuInfo]] = None


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


# @Route("/comfy/tasks")
@dataclass_json(letter_case=LetterCase.CAMEL, undefined=Undefined.EXCLUDE)
@dataclass
class GetComfyTasks(IReturn[ComfyTasksResponse], IGet):
    # @Validate(Validator="NotEmpty")
    # @Validate(Validator="ExactLength(32)")
    device_id: Optional[str] = None


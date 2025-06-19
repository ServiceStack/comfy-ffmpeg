# Filename: comfy_gateway_agent_node.py
# Place this file in your ComfyUI/custom_nodes/ directory,
# or in a subdirectory like ComfyUI/custom_nodes/my_utility_nodes/
# If in a subdirectory, ensure you have an __init__.py file in that subdirectory
# that exports the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

import io
import os
import uuid
import threading
import time
import requests
import json
import server # ComfyUI's server instance
import nodes
import subprocess
import logging
import traceback
import base64

from server import PromptServer
from folder_paths import get_user_directory, get_directory_by_type, models_dir
from comfyui_version import __version__

from .dtos import (
    RegisterComfyAgent, GetComfyAgentEvents, UpdateComfyAgent, UpdateWorkflowGeneration, GpuInfo, 
    CaptionArtifact, CompleteOllamaGenerateTask, GetOllamaGenerateTask
)
from servicestack.clients import UploadFile
from servicestack import JsonServiceClient, printdump, WebServiceException, ResponseStatus

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from .classifier import load_image_models, classify_image
from .imagehash import phash, dominant_color_hex

VERSION = 1
g_models = None
g_server_url = "http://localhost:7860"
g_headers_json={"Content-Type": "application/json"}
g_headers={}

g_current_config = {}  # Stores the active configuration for the global poller
g_logger_prefix = "[ComfyGatewayLogger]"
g_categories = []
g_assigned_prompts = []
g_uploaded_prompts = []
g_language_models = None

# --- Default Configuration for Autostart ---
# These values are used when ComfyUI starts, before any node instance takes control.
# The node's 'enabled' input will subsequently control the global task.
DEFAULT_AUTOSTART_ENABLED_ON_SERVER_LOAD = True 
DEFAULT_ENDPOINT_URL = "https://comfy-gateway.pvq.app"
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
BASE_URL= None
BEARER_TOKEN = None
DEVICE_ID = None
OLLAMA_BASE_URL = None
g_client = None

# --- End of Default Configuration ---

# Maintain a global dictionary of prompt_id mapping to client_id
g_pending_prompts = {}

g_models = None

def create_client():
    client = JsonServiceClient(BASE_URL)
    client.bearer_token = BEARER_TOKEN
    return client

def _log(message):
    """Helper method for logging from the global polling task."""
    print(f"{g_logger_prefix} {message}")

# Store the original method
original_send_sync = PromptServer.send_sync

# Define your interceptor function
def intercepted_send_sync(self, event, data, sid=None):
    # Your custom code to run before the event is sent
    _log(f"event={event}")

    if event == "executed" or event == "execution_success" or event == "status": 
        _log(json.dumps(data))
        # Do something with the execution data
        
    # Call the original method
    result = original_send_sync(self, event, data, sid)
    
    # Your custom code to run after the event is sent
    if event == "execution_success":
        _log("After execution_success event sent")
        prompt_id = data['prompt_id']
        if prompt_id in g_pending_prompts:
            client_id = g_pending_prompts[prompt_id]
            # call send_execution_success in a background thread
            threading.Thread(target=send_execution_success, args=(prompt_id, client_id), daemon=True).start()
    elif event == "execution_error":
        prompt_id = data['prompt_id']
        if prompt_id in g_pending_prompts:
            client_id = g_pending_prompts[prompt_id]
            _log("After execution_error event sent " + prompt_id)
            _log(json.dumps(data))
            exception_type = data['exception_type']
            exception_message = data['exception_message']
            traceback = data['traceback']
            # call send_execution_error in a background thread
            threading.Thread(target=send_execution_error, 
                args=(prompt_id, client_id, exception_type, exception_message, traceback), daemon=True).start()    
        
    
    return result

def remove_pending_prompt(prompt_id):
    if prompt_id in g_pending_prompts:
        del g_pending_prompts[prompt_id]

def send_execution_error(prompt_id, client_id, exception_type, exception_message, traceback):
    _log(f"send_execution_error: prompt_id={prompt_id}, client_id={client_id}")
    try:
        # only join first 5 lines of traceback
        stack_trace = "\n".join(traceback[:5])
        # split '.' and take last part
        message = f"{exception_type.split('.')[-1]}: {exception_message}"
        request = UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count(), 
            error=ResponseStatus(error_code=exception_type, message=message, stack_trace=stack_trace))
        g_client.post(request)
    except WebServiceException as ex:
        _log(f"Exception sending execution_error: {ex}")
        printdump(ex.response_status)
    except Exception as e:
        _log(f"Error sending execution_error: {e}")
    finally:
        remove_pending_prompt(prompt_id)

def send_execution_success(prompt_id, client_id):
    _log(f"send_execution_success: prompt_id={prompt_id}, client_id={client_id}")

    if prompt_id in g_uploaded_prompts:
        _log(f"prompt_id={prompt_id} already sent, skipping.")
        return

    try:
        result = PromptServer.instance.prompt_queue.get_history(prompt_id=prompt_id)
        prompt_data = result[prompt_id]
        outputs = prompt_data['outputs']
        status = prompt_data['status']
        _log(json.dumps(outputs))
        _log(json.dumps(status))

        # example outputs:
        # {"10": {"images": [{"filename": "ComfyUI_temp_pgpib_00001_.png", "subfolder": "", "type": "temp"}]}}

        #extract all image outputs
        artifacts = []
        for key, value in outputs.items():
            if 'images' in value:
                artifacts.extend(value['images'])
        # outputs = {"images": artifacts}
        _log(json.dumps(artifacts, indent=2))

        files = []
        for image in artifacts:
            dir = get_directory_by_type(image['type'])
            image_path = os.path.join(dir, image['subfolder'], image['filename'])
            #lowercase extension
            ext = image['filename'].split('.')[-1].lower()

            if (ext == "png" or ext == "jpg" or "jpeg" or "webp" or "gif" or "bmp" or "tiff"):
                with Image.open(image_path) as img:
                    image['width'] = img.width
                    image['height'] = img.height
                    # convert png to webp
                    if ext == "png":
                        quality = 90
                        buffer = io.BytesIO()
                        img.save(buffer, format='webp', quality=quality)
                        buffer.seek(0)
                        image_stream = buffer
                        image['filename'] = image['filename'].replace(".png", ".webp")
                        ext = "webp"
                    else:
                        image_stream=open(image_path, 'rb')

                    metadata = classify_image(g_models, g_categories, img, debug=True)
                    image.update(metadata)
                    image['phash'] = f"{phash(img)}"
                    image['color'] = dominant_color_hex(img)
    
            field_name = f"output_{len(files)}"
            files.append(UploadFile(
                field_name=field_name,
                file_name=image['filename'],
                content_type=f"image/{ext}",
                stream=image_stream
            ))

        request = UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count(), 
            outputs=json.dumps(outputs), 
            status=json.dumps(status))
        g_client.post_files_with_request(request, files)
        g_uploaded_prompts.append(prompt_id)
    except WebServiceException as ex:
        _log(f"Exception sending execution_success: {ex}")
        printdump(ex.response_status)
    except Exception as e:
        _log(f"Error sending execution_success: {e}")
    finally:
        remove_pending_prompt(prompt_id)
    
def urljoin(*args):
    trailing_slash = '/' if args[-1].endswith('/') else ''
    return "/".join([str(x).strip("/") for x in args]) + trailing_slash
    
# copied from PromptServer
def node_info(node_class):
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = {}
    info['input'] = obj_class.INPUT_TYPES()
    info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
    info['output'] = obj_class.RETURN_TYPES
    info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
    info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
    info['name'] = node_class
    info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
    info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
    info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
    info['category'] = 'sd'
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE is True:
        info['output_node'] = True
    else:
        info['output_node'] = False

    if hasattr(obj_class, 'CATEGORY'):
        info['category'] = obj_class.CATEGORY

    if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
        info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

    if getattr(obj_class, "DEPRECATED", False):
        info['deprecated'] = True
    if getattr(obj_class, "EXPERIMENTAL", False):
        info['experimental'] = True

    if hasattr(obj_class, 'API_NODE'):
        info['api_node'] = obj_class.API_NODE
    return info

def get_object_info():
    out = {}
    for x in nodes.NODE_CLASS_MAPPINGS:
        try:
            out[x] = node_info(x)
        except Exception:
            logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
            logging.error(traceback.format_exc())
    return out

def get_object_info_json():
    return json.dumps(get_object_info())

def get_object_info_json_from_url():
    json = requests.get(f"{g_server_url}/api/object_info").text
    return json

def listen_to_messages_poll():
    retry_secs = 5
    time.sleep(1)
    global g_client
    g_client = create_client()

    try:
        register_agent()
    except Exception as ex:
        _log(f"Error registering agent: {ex}")
        return

    while g_current_config["enabled"]:
        try:

            # check for pending prompts
            if len(g_assigned_prompts) > 0:
                _log(f"Processing {len(g_assigned_prompts)} pending prompts:")
                # iterate over pending prompts
                for ref in g_assigned_prompts:
                    _log(f"  {ref.client_id}:{ref.prompt_id}:{ref.api_prompt_url}")                    
                    # ignore if prompt is already in progress
                    if ref.prompt_id in g_pending_prompts.keys():
                        continue
                    if ref.client_id in g_pending_prompts:
                        continue
                    exec_prompt(ref.api_prompt_url)

                # clear pending prompts
                g_assigned_prompts.clear()

            send_update(sleep=0)

            _log("Polling for agent events")
            request = GetComfyAgentEvents(device_id=DEVICE_ID)

            response = g_client.get(request)
            retry_secs = 5
            if response.results is not None:
                event_names = [event.name for event in response.results]
                _log(f"Processing {len(response.results)} agent events: {','.join(event_names)}")
                for event in response.results:
                    if event.name == "Register":
                        register_agent()
                    elif event.name == "ExecWorkflow":
                        exec_prompt(event.args['url'])
                    elif event.name == "ExecOllama":
                        exec_ollama(event.args['model'], event.args['endpoint'], event.args['request'], event.args['replyTo'])
                    elif event.name == "CaptionImage":
                        caption_image(event.args['url'], event.args['model'])

        except Exception as ex:
            _log(f"Error connecting to {BASE_URL}: {ex}, retrying in {retry_secs}s")
            time.sleep(retry_secs)  # Wait before retrying
            retry_secs += 5 # Exponential backoff
            g_client = create_client() # Create new client to force reconnect

def get_queue_count():
    return PromptServer.instance.get_queue_info()['exec_info']['queue_remaining']

def send_update(sleep=0.1):
    if sleep > 0:
        time.sleep(sleep)
    try:
        current_queue = PromptServer.instance.prompt_queue.get_current_queue()
        queue_running = current_queue[0]
        queue_pending = current_queue[1]

        request = UpdateComfyAgent(device_id=DEVICE_ID, gpus=gpu_infos(),
            queue_count=len(queue_running) + len(queue_pending))
        request.queue_count = len(queue_running) + len(queue_pending)
        
        # get running generation ids (client_id) (max 20)
        request.running_generation_ids = [entry[3]['client_id'] for entry in queue_running 
            if len(entry[3] and entry[3]['client_id'] or '') == 32][:20]
        # get queued generation ids (client_id) (max 20)
        request.queued_generation_ids = [entry[3]['client_id'] for entry in queue_pending 
            if len(entry[3] and entry[3]['client_id'] or '') == 32][:20]
        
        _log(f"send_update: queue_count={request.queue_count}, running={request.running_generation_ids}, queued={request.queued_generation_ids}")
        g_client.post(request)
    except WebServiceException as ex:
        status = ex.response_status
        if status.error_code == "NotFound":
            _log("Device not found, reregistering")
            register_agent()
            return
        else:
            _log(f"Error sending update: {ex.message}\n{printdump(status)}")
    except Exception as e:
        _log(f"Error sending update: {e}")

def resolve_url(url):
    #if relative path, combine with BASE_URL
    if not url.startswith("http"):
        url = urljoin(BASE_URL, url)
    return url

def exec_prompt(url):

    if url is None:
        _log("exec_prompt: url is None")
        return

    # Get the server address - typically localhost when running within ComfyUI
    # server_address = PromptServer.instance.server_address
    # host, port = server_address if server_address else ("127.0.0.1", 7860)

    url = resolve_url(url)
    _log(f"exec_prompt GET: {url}")

    api_response = requests.get(url, headers=g_headers_json, timeout=30)
    if api_response.status_code != 200:
        _log(f"Error: {api_response.status_code} - {api_response.text}")
        return

    prompt_data = api_response.json()
    if 'client_id' not in prompt_data:
        _log("Error: No client_id in prompt data")
        return

    client_id = prompt_data['client_id']

    # check if client_id is a value in g_pending_prompts
    for key, value in g_pending_prompts.items():
        if value == client_id:
            prompt_id = key
            _log(f"exec_prompt: client_id={client_id} already in progress prompt_id={prompt_id}")
            g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
                queue_count=get_queue_count()))
            return

    _log(f"exec_prompt: /prompt client_id={client_id}")

    # Call the /prompt endpoint
    response = requests.post(
        f"{g_server_url}/prompt",
        json=prompt_data,
        headers=g_headers_json)

    if response.status_code == 200:
        result = response.json()
        prompt_id = result['prompt_id']
        _log(f"exec_prompt: /prompt OK prompt_id={prompt_id}, client_id={client_id}")
        _log(json.dumps(result))

        g_pending_prompts[prompt_id] = client_id
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count()))
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        _log(error_message)
        _log(json.dumps(prompt_data))
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, queue_count=get_queue_count(),
            error={"error_code": response.status_code, "message": response.text}))

def url_to_image(url):
    """Download an image from URL and return as PIL Image object"""
    try:
        response = requests.get(url) # Send GET request to download the image
        response.raise_for_status()  # Raises an HTTPError for bad responses        
        image = Image.open(io.BytesIO(response.content)) # Create PIL Image from the downloaded bytes
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def url_to_bytes(url):
    """Download an image from URL and return as PIL Image object"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def exec_ollama(model:str, endpoint:str, request:str, reply_to):
    error = None

    try:
        if g_language_models is None:
            error = ResponseStatus(error_code='Validation', message="Ollama is not available")
        elif model is None:
            error = ResponseStatus(error_code='Validation', message="model is None")
        elif endpoint is None:
            error = ResponseStatus(error_code='Validation', message="endpoint is None")
        elif request is None:
            error = ResponseStatus(error_code='Validation', message="request is None")
        elif reply_to is None:
            error = ResponseStatus(error_code='Validation', message="replyTo is None")
        elif model not in g_language_models:
            error = ResponseStatus(error_code='Validation', message=f"model {model} is not available")

        reply_url = resolve_url(reply_to)

        if error is not None:
            if reply_to is None:
                _log(f"exec_ollama: {error.error_code} {error.message}")
            else:
                body = {
                    'response_status': error
                }
                g_client.post_url(reply_url, body)
            return

        try:
            ollama_request = request
            if ollama_request.startswith('/') or ollama_request.startswith('http'):
                url = resolve_url(request)
                json = g_client.get_url(url, response_as=str)
                ollama_request = json
    
            # Send POST request to Ollama API
            ollama_url = f"{OLLAMA_BASE_URL}{endpoint}"
            _log(f"exec_ollama: POST {ollama_url}:")
            _log(f"{ollama_request[:100]}... ({len(ollama_request)})")
            response = requests.post(ollama_url, data=ollama_request, headers=g_headers_json, timeout=120)
            response.raise_for_status()
            
            # Parse response
            body = response.json()
            print(f"exec_ollama response: {body}")

            # Send response to replyTo URL
            g_client.post_url(reply_url, body)
            return

        except requests.exceptions.ConnectionError as e:
            _log("Error: Could not connect to Ollama API. Make sure Ollama is running on localhost:11434")
            error = ResponseStatus(error_code='ConnectionError', message=f"{e or 'Could not connect to Ollama API'}")
        except requests.exceptions.Timeout as e:
            _log("Error: Request timed out. The model might be taking too long to respond.")
            error = ResponseStatus(error_code='Timeout', message=f"{e or 'Request timed out'}")
        except requests.exceptions.RequestException as e:
            error = ResponseStatus(error_code='RequestException', message=f"{e or 'Error making request to Ollama API'}")
        except WebServiceException as e:
            error = e.responseStatus
        except Exception as e:
            error = ResponseStatus(error_code='Exception', message=f"{e}")

        body = {
            'responseStatus': {
                'errorCode': error.error_code,
                'message': error.message
            }
        }
        _log(f"exec_ollama error: {reply_url} {error.error_code} {error.message}")
        g_client.post_url(reply_url, body)
    except Exception as e:
        _log(f"Error executing Ollama: {e}")
        traceback.print_exc()

def ollama_generate(image_bytes, model, prompt):
    """
    Send an image to Ollama /api/generate API
    """
    # Ollama API endpoint
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    if not base64_image:
        return None
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    
    try:
        # Send POST request to Ollama API
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if 'response' in result:
            return result['response']
        else:
            print("Error: No 'response' field in API response")
            print(f"Full response: {result}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API. Make sure Ollama is running on localhost:11434")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model might be taking too long to respond.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

def caption_image(artifact_url, model):
    try:
        if g_language_models is None:
            _log(f"caption_image: g_language_models is None {OLLAMA_BASE_URL}")
            return    
        if artifact_url is None:
            _log("caption_image: url is None")
            return
        if model is None:
            _log("caption_image: model is None")
            return
        if model not in g_language_models:
            _log(f"caption_image: model {model} is not available")
            return
        
        url = resolve_url(artifact_url)
        _log(f"caption_image ({model}) GET: {url}")

        image_bytes = url_to_bytes(url)
        if image_bytes is None:
            return
        
        request = CaptionArtifact(device_id=DEVICE_ID, artifact_url=artifact_url)
        request.caption = ollama_generate(image_bytes, model, "A caption of this image: ")
        request.description = ollama_generate(image_bytes, model, "A detailed description of this image: ")
        
        _log(f"caption_image caption: {request.caption}\n{request.description}")
        g_client.post(request)

    except Exception as e:
        _log(f"Error captioning image: {e}")
        traceback.print_exc()

def on_prompt_handler(json_data):
    if g_current_config["enabled"]:
        # run send_update once in background thread
        threading.Thread(target=send_update, daemon=True).start()
    return json_data

def gpu_infos():
    #get info of gpus from $nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader,nounits
    # example output: 0, 16303, 13991, 1858
    gpus = []
    output = ''
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'])
        lines = output.decode('utf-8').strip().split('\n')
        for line in lines:
            index, name, total, free, used = line.split(',')
            gpu = GpuInfo(index=int(index),name=name.strip(),total=int(total),free=int(free),used=int(used))
            gpus.append(gpu)
    except Exception as e:
        _log(f"Error getting GPU info: {e}\n{output}")
    return gpus

def gpus_as_jsv():
    gpus = gpu_infos()
    # complex types on the query string need to be sent with JSV format
    ret = ','.join(['{' + f"index:{gpu.index},name:\"{gpu.name}\",total:{gpu.total},free:{gpu.free},used:{gpu.used}" + '}' for gpu in gpus])
    return ret

def register_agent():
    # get workflows from user/default/workflows
    user_dir = get_user_directory()
    workflows_dir = os.path.join(user_dir, "default", "workflows")
    workflows = []
    # exclude .json starting with '.'
    if os.path.exists(workflows_dir):
        workflows = [f for f in os.listdir(workflows_dir) if f.endswith(".json") and not f.startswith(".")]

    object_info_json = get_object_info_json()

    object_info_file = UploadFile(
        field_name="object_info",
        file_name="object_info.json",
        content_type="application/json",
        stream=io.BytesIO(object_info_json.encode('utf-8')))
    
    global g_language_models
    g_language_models = None
    if OLLAMA_BASE_URL is not None:
        try:
            g_language_models = []
            # Check if Ollama is running by hitting the base endpoint with a reasonable timeout
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract models from the response
            models = data.get('models', [])
            
            print(f"✅ Ollama is running with {len(models)} installed models")
            for i, model in enumerate(models, 1):
                name = model.get('name')
                if name is not None:
                    g_language_models.append(name)
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print("Make sure Ollama is running and accessible")
            return False, None
        except requests.exceptions.Timeout:
            print(f"❌ Request to {OLLAMA_BASE_URL} timed out")
            return False, None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to Ollama: {e}")
            return False, None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False, None

    response = g_client.post_file_with_request(
        request=RegisterComfyAgent(
            device_id=DEVICE_ID,
            version=VERSION,
            workflows=workflows,
            gpus=gpu_infos(),
            queue_count=get_queue_count(),
            language_models=g_language_models,
        ),
        file=object_info_file)

    _log(f"Registered device with {BASE_URL}")
    printdump(response)

    # check if response.categories is an array with items
    if isinstance(response.categories, list):
        global g_categories
        g_categories = response.categories

    _log(f"Pending prompts: {len(response.pending_prompts)}")
    if (isinstance(response.pending_prompts, list) and len(response.pending_prompts) > 0):
        g_assigned_prompts = response.pending_prompts


def setup_connection():
    """
    This function is called when the ComfyUI server has loaded or the module is imported.
    """
    global g_current_config

    _log("Setting up global polling task.")

    # Initialize g_current_config with defaults.
    # The 'enabled' key here determines if it autostarts.
    g_current_config = {
        "enabled": DEFAULT_AUTOSTART_ENABLED_ON_SERVER_LOAD,
        "url": DEFAULT_ENDPOINT_URL,
        "interval": DEFAULT_POLL_INTERVAL_SECONDS,
        "req_timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS
    }

    PromptServer.instance.add_on_prompt_handler(on_prompt_handler)

    if g_current_config["enabled"]:

        try:
            # register_agent()

            # listen to messages in a background thread
            t = threading.Thread(target=listen_to_messages_poll, daemon=True)
            t.start()
        except Exception as e:
            _log(f"Error registering device: {e}")
            raise e

    
    else:
        _log("Autostart is disabled by default server configuration.")

# --- ComfyUI Node Definition ---
class ComfyGatewayAgentNode:
    NODE_NAME = "ComfyGatewayAgentNode"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "manage_global_polling_from_node"
    OUTPUT_NODE = True
    CATEGORY = "ComfyGateway"

    @classmethod
    def INPUT_TYPES(cls):
        # Use global defaults for the node's default inputs
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": DEFAULT_AUTOSTART_ENABLED_ON_SERVER_LOAD, "label_on": "POLLING ENABLED", "label_off": "POLLING DISABLED"}),
                "endpoint_url": ("STRING", {"default": DEFAULT_ENDPOINT_URL}),
                "poll_interval_seconds": ("INT", {"default": DEFAULT_POLL_INTERVAL_SECONDS, "min": 1, "max": 3600, "step": 1}),
                "request_timeout_seconds": ("INT", {"default": DEFAULT_REQUEST_TIMEOUT_SECONDS, "min": 5, "max": 3600, "step": 1}),
            },
            "optional": {
                "trigger_restart": ("*",) 
            }
        }

    def __init__(self):
        self._node_log_prefix_str = f"[{self.NODE_NAME} id:{hex(id(self))[-4:]}]"
        self._log("Node instance initialized. This node controls the global polling task.")

    def _log(self, message):
        """Helper method for logging from the node instance."""
        print(f"{self._node_log_prefix_str} {message}")

# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    ComfyGatewayAgentNode.NODE_NAME: ComfyGatewayAgentNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    ComfyGatewayAgentNode.NODE_NAME: "Comfy Gateway Agent (Global)"
}

# --- Autostart Logic ---

# Check COMFY_GATEWAY environment variable for BASE_URL and BEARER_TOKEN configuration:
# BEARER_TOKEN@BASE_URL
if "COMFY_GATEWAY" in os.environ:
    if "@" in os.environ['COMFY_GATEWAY']:
        bearer_token, base_url = os.environ['COMFY_GATEWAY'].split("@")
        BASE_URL = base_url
        BEARER_TOKEN = bearer_token
        g_client = create_client()
        hidden_token = BEARER_TOKEN[:3] + ("*" * 3) + BEARER_TOKEN[-2:]
        _log(f"ComfyGateway BASE_URL: {BASE_URL}, BEARER_TOKEN: {hidden_token}")

        try:
            # Replace the original method with your interceptor
            PromptServer.send_sync = intercepted_send_sync

            # Read device ID from users/device-id
            device_id_path = os.path.join(get_user_directory(), "device-id")
            # check if file exists

            if os.path.isfile(device_id_path):
                with open(device_id_path) as f:
                    DEVICE_ID = f.read().strip()
                _log(f"DEVICE_ID: {DEVICE_ID}")
            else:
                # write device id
                _log(f"Generating Device ID at {device_id_path}")
                DEVICE_ID = uuid.uuid4().hex
                with open(device_id_path, "w") as f:
                    f.write(DEVICE_ID)

        except IOError:
            DEVICE_ID = uuid.uuid4().hex
            _log(f"Failed to read device ID from {device_id_path}. Generating a new one: {DEVICE_ID}")

        try:
            if "OLLAMA_BASE_URL" in os.environ:
                OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
            g_models = load_image_models(models_dir=models_dir, debug=True)
            g_headers["User-Agent"] = g_headers_json["User-Agent"] = f"comfy-ffmpeg/{DEVICE_ID}/{__version__}"
            try:
                setup_connection()
            except Exception:
                logging.error("[ERROR] Could not connect to ComfyGateway.")
                logging.error(traceback.format_exc())
        except Exception:
            logging.error("[ERROR] Could not load models.")
            logging.error(traceback.format_exc())
    else:
        _log(f"Warning: COMFY_GATEWAY environment variable is not in the correct format. Expected 'BEARER_TOKEN@BASE_URL', got '{os.environ['COMFY_GATEWAY']}'.")

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

from server import PromptServer
from folder_paths import get_user_directory, get_directory_by_type, models_dir

from .dtos import *
from servicestack.clients import UploadFile
from servicestack import JsonServiceClient, printdump

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from .classifier import classify_image_rating, classify_image_tags, load_clip, detect_objects

g_current_config = {}  # Stores the active configuration for the global poller
g_logger_prefix = "[ComfyGatewayLogger]"
g_primary_tags = []
g_assigned_prompts = []

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
g_client = None

# --- End of Default Configuration ---

# Maintain a global dictionary of prompt_id mapping to client_id
g_pending_prompts = {}

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
        _log(json.dumps(artifacts))

        artiract_objects = {}
        image_paths = [os.path.join(get_directory_by_type(image['type']), image['subfolder'], image['filename']) for image in artifacts]
        try:
            start_time = time.time()
            artiract_objects = detect_objects(os.path.join(models_dir, "nsfw"), image_paths)
            elapsed_time = time.time() - start_time
            print(f"detect_objects took {elapsed_time:.2f}s", image_paths, len(artiract_objects))
            printdump(artiract_objects)
        except Exception as e:
            print(f"Error in detect_objects: {e}")

        try:
            start_time = time.time()
            device, model, preprocess = load_clip()
            elapsed_time = time.time() - start_time
            print(f"Loaded ratings clip model in {elapsed_time:.2f}s")
        except Exception as e:
            print(f"Error loading clip model: {e}")

        files = []
        for image in artifacts:
            dir = get_directory_by_type(image['type'])
            image_path = os.path.join(dir, image['subfolder'], image['filename'])
            #lowercase extension
            ext = image['filename'].split('.')[-1].lower()

            ratings = None
            tags = None
            # convert png to webp
            if ext == "png":
                with Image.open(image_path) as img:
                    quality = 90
                    buffer = io.BytesIO()
                    img.save(buffer, format='webp', quality=quality)
                    buffer.seek(0)
                    image_stream = buffer
                    image['filename'] = image['filename'].replace(".png", ".webp")
                    ext = "webp"
                    image['width'] = img.width
                    image['height'] = img.height
                    if device is not None:
                        start_time = time.time()
                        ratings = classify_image_rating(img, device, model, preprocess)
                        if len(g_primary_tags) > 0:
                            tags = classify_image_tags(img, g_primary_tags, device, model, preprocess)
                        elapsed_time = time.time() - start_time
                        print(f"Classified image rating and tags in {elapsed_time:.2f}s")
            else:
                if (ext == "jpg" or "jpeg" or "webp" or "gif" or "bmp" or "tiff") and device is not None:
                    with Image.open(image_path) as img:
                        ratings = classify_image_rating(img, device, model, preprocess)
                        if len(g_primary_tags) > 0:
                            tags = classify_image_tags(img, g_primary_tags, device, model, preprocess)
                        image['width'] = img.width
                        image['height'] = img.height
                image_stream=open(image_path, 'rb')

            if ratings is not None:
                image['ratings'] = ratings
            if tags is not None:
                image['tags'] = tags
            if image_path in artiract_objects:
                image['objects'] = artiract_objects[image_path]
            else:
                print(f"No objects found for {image_path}")

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
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
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

def listen_to_messages_poll():
    retry_secs = 5
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

            _log("Polling for agent events")
            request = GetComfyAgentEvents(device_id=DEVICE_ID)
            # get prompt ids of queued prompts
            current_queue = PromptServer.instance.prompt_queue.get_current_queue()
            queue_running = current_queue[0]
            queue_pending = current_queue[1]
            
            # get running generation ids (client_id) (max 20)
            request.running_generation_ids = [entry[3]['client_id'] for entry in queue_running 
                if len(entry[3] and entry[3]['client_id'] or '') == 32][:20]

            # get queued generation ids (client_id) (max 20)
            request.queued_generation_ids = [entry[3]['client_id'] for entry in queue_pending 
                if len(entry[3] and entry[3]['client_id'] or '') == 32][:20]
            request.queue_count = len(queue_running) + len(queue_pending)

            _log(f"GetComfyAgentEvents count={request.queue_count}, running={request.running_generation_ids}, queued={request.queued_generation_ids}")

            response = g_client.get(request)
            retry_secs = 5
            if response.results is not None:
                event_names = [event.name for event in response.results]
                _log(f"Processing {len(response.results)} agent events: {','.join(event_names)}")
                for event in response.results:
                    if event.name == "Register":
                        register_agent()
                    if event.name == "ExecWorkflow":
                        exec_prompt(event.args['url'])

        except Exception as ex:
            _log(f"Error connecting to {BASE_URL}: {ex}, retrying in {retry_secs}s")
            time.sleep(retry_secs)  # Wait before retrying
            retry_secs += 5 # Exponential backoff
            client = create_client() # Create new client to force reconnect

def get_queue_count():
    return PromptServer.instance.get_queue_info()['exec_info']['queue_remaining']

def send_update():
    time.sleep(0.1)
    try:
        request = UpdateComfyAgent(
            device_id=DEVICE_ID,
            queue_count=get_queue_count(),
            gpus=gpu_infos())
        _log(f"send_update: queue_count={request.queue_count}")
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

def exec_prompt(url):

    if url is None:
        _log("exec_prompt: url is None")
        return

    # Get the server address - typically localhost when running within ComfyUI
    # server_address = PromptServer.instance.server_address
    # host, port = server_address if server_address else ("127.0.0.1", 7860)
    server_url = "http://localhost:7860"
    headers={"Content-Type": "application/json"}

    #if relative path, combine with BASE_URL
    if not url.startswith("http"):
        url = urljoin(BASE_URL, url)

    _log(f"exec_prompt GET: {url}")

    api_response = requests.get(url, headers=headers, timeout=30)
    if api_response.status_code != 200:
        _log(f"Error: {api_response.status_code} - {api_response.text}")
        return

    prompt_data = api_response.json()
    if 'client_id' not in prompt_data:
        _log(f"Error: No client_id in prompt data")
        return

    client_id = prompt_data['client_id']
    gpus = gpu_infos()

    # check if client_id is a value in g_pending_prompts
    for key, value in g_pending_prompts.items():
        if value == client_id:
            prompt_id = key
            _log(f"exec_prompt: client_id={client_id} already in progress prompt_id={prompt_id}")
            g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
                queue_count=get_queue_count(), gpus=gpus))
            return

    _log(f"exec_prompt: /prompt client_id={client_id}")

    # Call the /prompt endpoint
    response = requests.post(
        f"{server_url}/prompt",
        json=prompt_data,
        headers=headers)

    if response.status_code == 200:
        result = response.json()
        prompt_id = result['prompt_id']
        _log(f"exec_prompt: /prompt OK prompt_id={prompt_id}, client_id={client_id}")
        _log(json.dumps(result))

        g_pending_prompts[prompt_id] = client_id
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count(), gpus=gpus))
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        _log(error_message)
        _log(json.dumps(prompt_data))
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, 
            queue_count=get_queue_count(), gpus=gpus,
            error={"error_code": response.status_code, "message": response.text}))


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

    object_info = get_object_info()
    object_info_json = json.dumps(object_info)

    object_info = UploadFile(
        field_name="object_info",
        file_name="object_info.json",
        content_type="application/json",
        stream=io.BytesIO(object_info_json.encode('utf-8')))
    
    response = g_client.post_file_with_request(
        request=RegisterComfyAgent(
            device_id=DEVICE_ID,
            workflows=workflows,
            gpus=gpus_as_jsv(),
            queue_count=get_queue_count()
        ),
        file=object_info)

    _log(f"Registered device with {BASE_URL}")
    printdump(response)

    # check if response.tags is an array with items
    if isinstance(response.tags, list):
        global g_primary_tags
        g_primary_tags = response.tags

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
            register_agent()

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
            setup_connection()
        except Exception as e:
            _log(f"Could not connect to ComfyGateway: {e}. ")
    else:
        _log(f"Warning: COMFY_GATEWAY environment variable is not in the correct format. Expected 'BEARER_TOKEN@BASE_URL', got '{os.environ['COMFY_GATEWAY']}'.")

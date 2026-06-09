import json
import re
import os
from pathlib import Path
from functools import reduce

import gradio as gr
from fastapi import params
from matplotlib import image
import piexif

from ch_lib import util, extra_resources
from modules import script_callbacks, extra_networks, prompt_parser, processing, sd_models, infotext_utils, shared, images
from modules import scripts_postprocessing
import networks # extensions-builtin\sd_forge_lora\networks.py
try:
    from backend.args import dynamic_args
except ModuleNotFoundError:
    dynamic_args = None

try:
    import modules.processing_scripts.comments as comments
except ModuleNotFoundError:
    comments = None

re_prompt = re.compile(r"^(?!.+\sneg(?:ative)?)(.+\s)prompt(\s\S+)?$")
re_negative_prompt = re.compile(r"^(.+\s)neg(?:ative)?\sprompt(\s\S+)?$")
re_checkpoint = re.compile(r"^(?!Hires).+\scheckpoint(?:\s\S+)?$")

# Checkbox state set by ScriptPostprocessingCivitaiHelper.process() before each save.
_preserve_infotext = True


class ScriptPostprocessingCivitaiHelper(scripts_postprocessing.ScriptPostprocessing):
    """Adds a 'Preserve Original Infotext' checkbox to the Extras tab."""

    name = "Civitai Helper"
    order = 9000  # appear after upscalers / face restoration

    def ui(self):
        with gr.Accordion("Civitai Helper", open=True, elem_id="ch_extras_accordion"):
            preserve = gr.Checkbox(
                label="Preserve Original Infotext",
                value=True,
                elem_id="ch_extras_preserve_infotext",
                info="Merge the source image's generation parameters into the saved upscale metadata. Requires 'Automatically add resource metadata' in Settings › Civitai Helper.",
            )
        return {"preserve_infotext": preserve}

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        global _preserve_infotext
        _preserve_infotext = args.get("preserve_infotext", True)


def add_resource_metadata(params):
    util.printD("Adding resource metadata...")
    if not util.get_opts("ch_image_metadata") or 'parameters' not in params.pnginfo:
        util.printD("Option not enabled or parameters missing. Skipping resource metadata...")
        return
    # Postprocessing saves have params.p == None; handled by add_postprocessing_metadata
    if params.p is None:
        util.printD("No processing object found. Assuming postprocessing save and skipping resource metadata...")
        return

    # StableDiffusionProcessing
    sd_processing = params.p
    # CheckpointInfo
    sd_checkpoint_info = sd_models.get_closet_checkpoint_match(sd_processing.sd_model_name)

    civitai_resource_list = []

    def add_civitai_resource(base_file_path, weight=None, type_name=None):
        try:
            # Read civitai metadata from previously generated info file
            file_path = Path(base_file_path).with_suffix(".civitai.info")
            with open(file_path, 'r') as file:
                civitai_info = json.load(file)
                resource_data = {}
                resource_data["type"] = type_name if type_name is not None else civitai_info["model"]["type"].lower()
                if resource_data["type"] in ["locon", "loha"]:
                    resource_data["type"] = "lycoris"
                if weight is not None:
                    resource_data["weight"] = weight
                resource_data["modelVersionId"] = civitai_info["id"]
                resource_data["modelName"] = civitai_info["model"]["name"]
                resource_data["modelVersionName"] = civitai_info["name"]
                civitai_resource_list.append(resource_data)
        except FileNotFoundError:
            util.printD(f"Warning: '{file_path}' not found. Did you forget to scan?")
        except Exception as e:
            util.printD(f"Civitai info error: {e}")

    checkpoint_set = set([sd_checkpoint_info.name])

    # Parse infotext first so it can serve as a fallback when params.p is a
    # minimal object (e.g. when saving via the UI save button rather than during
    # generation).  In that case attributes like prompt/negative_prompt/
    # extra_network_data are not available on the processing object.
    generation_parameters = infotext_utils.parse_generation_parameters(params.pnginfo['parameters'])

    sd_prompt = getattr(sd_processing, 'prompt', generation_parameters.get('Prompt', ''))
    sd_negative_prompt = getattr(sd_processing, 'negative_prompt', generation_parameters.get('Negative prompt', ''))
    sd_steps = getattr(sd_processing, 'steps', int(generation_parameters.get('Steps', 20) or 20))

    prompt_list = [[sd_prompt, sd_steps, True], [sd_negative_prompt, sd_steps, False]]

    # Get extra_network_data from the processing object when available, otherwise
    # fall back to parsing it from the prompt text (UI save button path).
    raw_extra_network_data = getattr(sd_processing, 'extra_network_data', None)
    if raw_extra_network_data is not None:
        extra_network_data = raw_extra_network_data.values()
    else:
        prompt_stripped = (comments.strip_comments(sd_prompt) if comments else sd_prompt).strip()
        _, parsed_network_data = extra_networks.parse_prompt(prompt_stripped)
        extra_network_data = list(parsed_network_data.values())

    # Add hires. fix data
    if isinstance(sd_processing, processing.StableDiffusionProcessingTxt2Img) and sd_processing.enable_hr:
        if sd_processing.hr_checkpoint_name is not None:
            checkpoint_set.add(sd_processing.hr_checkpoint_info.name)
        prompt_list += [[sd_processing.hr_prompt, sd_processing.hr_second_pass_steps, True], [sd_processing.hr_negative_prompt, sd_processing.hr_second_pass_steps, False]]
        extra_network_data = list(extra_network_data) + list(sd_processing.hr_extra_network_data.values())

    # TODO: img2img/upscale - add original image resources

    # Read prompt/generation data from other extensions, e.g., ADetailer, μDDetailer
    for key, value in generation_parameters.items():
        prompt_match = re_prompt.search(key)
        negative_prompt_match = re_negative_prompt.search(key)

        if prompt_match is not None or negative_prompt_match is not None:
            prompt = value
            is_positive = bool(prompt_match)
            match = prompt_match if is_positive else negative_prompt_match

            prefix, suffix = match.group(1, 2)
            steps_key = f"{prefix}steps{suffix if suffix is not None else ''}"
            steps = int(generation_parameters[steps_key]) if steps_key in generation_parameters and int(generation_parameters[steps_key]) != 0 else sd_processing.steps

            prompt_list += [[prompt, steps, is_positive]]

            comments_stripped = (comments.strip_comments(prompt) if comments else prompt).strip()
            _, found_network_data = extra_networks.parse_prompt(comments_stripped)
            extra_network_data = list(extra_network_data) + list(found_network_data.values())

        elif re_checkpoint.search(key) is not None:
            checkpoint_set.add(value)

    # Add checkpoint metadata
    for checkpoint_name in checkpoint_set:
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
        if checkpoint_info is not None:
            add_civitai_resource(Path(checkpoint_info.filename).absolute())
        else:
            util.printD(f"Error: '{checkpoint_name}' not found.")

    # Collect lora weights, skip duplicates
    extra_network_weights = {}
    if len(extra_network_data) > 0 if isinstance(extra_network_data, list) else any(extra_network_data):
        for extra_network_params in reduce(lambda list1, list2: list1 + list2, extra_network_data):
            extra_network_name = extra_network_params.positional[0]
            if extra_network_name not in extra_network_weights:
                te_multiplier = float(extra_network_params.positional[1]) if len(extra_network_params.positional) > 1 else 1.0
                extra_network_weights[extra_network_name] = te_multiplier

    # Add lora metadata
    for extra_network_name, te_multiplier in extra_network_weights.items():
        network_on_disk = networks.available_network_aliases.get(extra_network_name, None)
        if network_on_disk is not None:
            add_civitai_resource(Path(network_on_disk.filename).absolute(), te_multiplier)
        else:
            util.printD(f"Error: '{extra_network_name}' alias not found.")

    # Get embedding file paths
    embedding_dir = dynamic_args['embedding_dir'] if dynamic_args else getattr(shared.cmd_opts, 'embeddings_dir', None)
    embed_filepaths = {}
    try:
        for dirpath, _, filenames in os.walk(embedding_dir, followlinks=True) if embedding_dir else []:
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.stat().st_size != 0 and filepath.suffix.upper() in ['.BIN', '.PT', '.SAFETENSORS']:
                    embed_filepaths[filepath.stem.strip().lower()] = filepath.absolute()
    except Exception as e:
        util.printD(f"Embedding directory error: {e}")

    # Add textual inversion embed metadata
    if len(embed_filepaths) > 0:
        embed_weights = {}
        try:
            embed_regex = re.compile(r"(?:^|[\s,.])(" + '|'.join(re.escape(embed_name) for embed_name in embed_filepaths.keys()) + r")(?:$|[\s,.])", re.IGNORECASE | re.MULTILINE)

            for prompt, steps, is_positive in prompt_list:
                # parse all special prompt rules
                comments_stripped = (comments.strip_comments(prompt) if comments else prompt).strip()
                extra_networks_stripped, _ = extra_networks.parse_prompt(comments_stripped)
                if is_positive:
                    _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([extra_networks_stripped])
                else:
                    prompt_flat_list = [extra_networks_stripped]
                prompt_edit_schedule = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
                prompts = [text for step, text in reduce(lambda list1, list2: list1 + list2, prompt_edit_schedule)]
                for scheduled_prompt in prompts:
                    # calculate attention weights
                    for text, weight in prompt_parser.parse_prompt_attention(scheduled_prompt):
                        for match in embed_regex.findall(text):
                            # store final weight of embedding in dictionary
                            embed_weights[match.lower()] = weight
        except Exception as e:
            util.printD(f"Error parsing prompt for embeddings: {e}")

        # add final weights for embeddings
        for embed_name, weight in embed_weights.items():
            add_civitai_resource(embed_filepaths[embed_name], weight, "embed")

    # Find upscalers that have civitai info files
    upscaler_civitai_paths = {}
    for upscaler_data in shared.sd_upscalers:
        if not upscaler_data.data_path:
            continue
        info_path = Path(upscaler_data.data_path).with_suffix(".civitai.info")
        if info_path.is_file():
            upscaler_civitai_paths[upscaler_data.name] = upscaler_data.data_path

    # Add upscaler metadata if any tracked upscaler was used
    if upscaler_civitai_paths:
        upscalers_used = set()
        # Check hires fix upscaler from processing object
        if isinstance(sd_processing, processing.StableDiffusionProcessingTxt2Img) and sd_processing.enable_hr:
            if sd_processing.hr_upscaler:
                upscalers_used.add(sd_processing.hr_upscaler)
        # Check generation parameters for any other upscaler references.
        # Some scripts (e.g. Ultimate SD Upscale) save the upscaler name without
        # file extension under arbitrary key names, so match by value directly.
        for key, value in generation_parameters.items():
            if isinstance(value, str) and value in upscaler_civitai_paths:
                upscalers_used.add(value)
        for upscaler_name in upscalers_used:
            if upscaler_name in upscaler_civitai_paths:
                add_civitai_resource(upscaler_civitai_paths[upscaler_name], type_name="upscaler")

    # Merge any manually-added resources from the txt2img/img2img accordion
    if extra_resources.get_enabled():
        for extra in extra_resources.get():
            if extra not in civitai_resource_list:
                civitai_resource_list.append(extra)

    if len(civitai_resource_list) > 0:
        params.pnginfo['parameters'] += f", Civitai resources: {json.dumps(civitai_resource_list, separators=(',', ':'))}"


def add_postprocessing_metadata(params):
    """Handle Extras-tab upscale saves: merge infotext and add upscaler civitai resources."""
    util.printD("Adding postprocessing metadata...")

    if not util.get_opts("ch_image_metadata"):
        util.printD("Option not enabled or parameters missing. Skipping resource metadata...")
        return

    if not _preserve_infotext:
        util.printD("Preserve Original Infotext is disabled. Skipping postprocessing metadata...")
        return

    # Only for postprocessing saves (no processing object, postprocessing key present)
    if params.p is not None:
        util.printD("No processing object found. Assuming generation save and skipping postprocessing metadata...")
        return
    postprocessing_infotext = params.pnginfo.get("postprocessing")
    if not postprocessing_infotext:
        util.printD("No postprocessing infotext found. Skipping postprocessing metadata...")
        return

    # 1. Append postprocessing info to the parameters string so it travels with the image
    if "parameters" not in params.pnginfo:
        params.pnginfo["parameters"] = postprocessing_infotext

    metadata_match = re.search(
        r'Civitai metadata:\s*(\{.*\})',
        params.pnginfo["parameters"]
    )

    metadata = {}

    if metadata_match:
        metadata = json.loads(metadata_match.group(1))

    ori_seed = metadata.get("seed")

    params.pnginfo["parameters"] = re.sub(
        r'Seed:\s*-?\d+',
        f'Seed: {ori_seed}',
        params.pnginfo["parameters"]
    )

    metadata_resources = metadata.get("resources", [])

    match = re.search(r'Civitai resources:\s*(\[[^\]]*\])', params.pnginfo["parameters"])

    if match:
        civitai_resource_list = json.loads(match.group(1))
    else:
        civitai_resource_list = []

    civitai_resource_list.extend(metadata_resources)

    upscaler_civitai_paths = {}
    for upscaler_data in shared.sd_upscalers:
        if not upscaler_data.data_path:
            continue
        info_path = Path(upscaler_data.data_path).with_suffix(".civitai.info")
        if info_path.is_file():
            upscaler_civitai_paths[upscaler_data.name] = upscaler_data.data_path

    if upscaler_civitai_paths:
        # The infotext format is: Postprocess upscaler: "Name", Postprocess upscaler 2: "Name"
        for m in re.finditer(r'Postprocess upscaler(?:\s2)?:\s*("?)([^",]+)\1', postprocessing_infotext):
            upscaler_name = m.group(2)
            util.printD(f"Found upscaler in postprocessing infotext: '{upscaler_name}'")
            if upscaler_name not in upscaler_civitai_paths:
                continue
            try:
                file_path = Path(upscaler_civitai_paths[upscaler_name]).with_suffix(".civitai.info")
                with open(file_path, 'r') as f:
                    civitai_info = json.load(f)
                resource = {
                    "type": "upscaler",
                    "modelVersionId": civitai_info["id"],
                    "modelName": civitai_info["model"]["name"],
                    "modelVersionName": civitai_info["name"],
                }
                if resource not in civitai_resource_list:
                    civitai_resource_list.append(resource)
            except Exception as e:
                util.printD(f"Civitai upscaler info error: {e}")

    if civitai_resource_list:
        resources_json = json.dumps(civitai_resource_list, separators=(',', ':'))
        if match:
            params.pnginfo["parameters"] = re.sub(
                r'Civitai resources:\s*\[[^\]]*\]',
                f'Civitai resources: {resources_json}',
                params.pnginfo["parameters"]
            )
        else:
            params.pnginfo["parameters"] += f", Civitai resources: {resources_json}"

    # Also update the "postprocessing" key so JPG/WebP EXIF gets the merged content
    # (for PNG all keys are written; for JPG/WebP only "postprocessing" → EXIF UserComment)
    params.pnginfo["postprocessing"] = params.pnginfo["parameters"]


# def verify_postprocessing_saved(params):
#     """Debug: read back the saved file and print what's actually in it."""
#     if params.p is not None:
#         return
#     try:
#         from PIL import Image
#         saved_path = getattr(params.image, "already_saved_as", None) or params.filename
#         img = Image.open(saved_path)

#         geninfo = images.read_info_from_image(img)[0]

#         images.save_image_with_geninfo(img, geninfo, saved_path)
#     except Exception as e:
#         util.printD(f"[verify] error reading back file: {e}")


script_callbacks.on_before_image_saved(add_resource_metadata)
script_callbacks.on_before_image_saved(add_postprocessing_metadata)
# script_callbacks.on_image_saved(verify_postprocessing_saved)


# ---------------------------------------------------------------------------
# Auto-detect resources from pasted infotext (e.g. "Send to img2img")
# ---------------------------------------------------------------------------

# Track modelVersionIds that were added automatically so we can replace them
# when the next image is sent, while keeping manually-added resources intact.
_last_auto_resource_ids: set = set()


def _load_civitai_info(base_path) -> dict | None:
    """Return parsed .civitai.info for *base_path* (any extension), or None."""
    try:
        info_path = Path(base_path).with_suffix(".civitai.info")
        if not info_path.is_file():
            return None
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        util.printD(f"civitai.info read error for {base_path}: {e}")
        return None


def on_infotext_pasted(infotext: str, params: dict):
    """Auto-populate the extra resources list from the checkpoint and LoRAs
    referenced in pasted infotext (fired when an image is sent to any tab)."""

    if not util.get_opts("ch_image_metadata"):
        return

    global _last_auto_resource_ids

    # Remove previously auto-detected resources, keep manually-added ones.
    if _last_auto_resource_ids:
        kept = [r for r in extra_resources.get()
                if r.get("modelVersionId") not in _last_auto_resource_ids]
        extra_resources.clear()
        for r in kept:
            extra_resources.add(r)
    _last_auto_resource_ids = set()

    detected = []

    # 1. Checkpoint -----------------------------------------------------------
    model_name = params.get("Model")
    if model_name:
        util.printD(f"model name in params: '{model_name}'")
        checkpoint_info = sd_models.get_closet_checkpoint_match(model_name)
        if checkpoint_info:
            civitai_info = _load_civitai_info(checkpoint_info.filename)
            if civitai_info:
                detected.append({
                    "type": civitai_info["model"]["type"].lower(),
                    "modelVersionId": civitai_info["id"],
                    "modelName": civitai_info["model"]["name"],
                    "modelVersionName": civitai_info["name"],
                })
    else:
        util.printD("No model name found in params.")

    # 2. LoRAs in the positive prompt -----------------------------------------
    prompt = params.get("Prompt", "")
    for m in re.finditer(r'<lora:([^:>]+)(?::([^>]+))?>', prompt):
        lora_name = m.group(1)
        try:
            weight = float(m.group(2)) if m.group(2) else 1.0
        except ValueError:
            weight = 1.0

        network_on_disk = (networks.available_network_aliases.get(lora_name)
                           or networks.available_network_aliases.get(lora_name.lower()))
        if not network_on_disk:
            continue
        civitai_info = _load_civitai_info(network_on_disk.filename)
        if not civitai_info:
            continue
        lora_type = civitai_info["model"]["type"].lower()
        if lora_type in ("locon", "loha"):
            lora_type = "lycoris"
        detected.append({
            "type": lora_type,
            "weight": weight,
            "modelVersionId": civitai_info["id"],
            "modelName": civitai_info["model"]["name"],
            "modelVersionName": civitai_info["name"],
        })

    if not detected:
        return

    existing_ids = {r.get("modelVersionId") for r in extra_resources.get()}
    for resource in detected:
        ver_id = resource.get("modelVersionId")
        _last_auto_resource_ids.add(ver_id)
        if ver_id not in existing_ids:
            extra_resources.add(resource)
            existing_ids.add(ver_id)
            util.printD(f"Auto-added resource: {resource['modelName']} ({resource['type']})")


script_callbacks.on_infotext_pasted(on_infotext_pasted)

""" -*- coding: UTF-8 -*-
Adds a "Civitai Helper" accordion to the txt2img and img2img tabs.

Users can paste Civitai model URLs, fetch metadata, and build a temporary list
of resources that will be appended to the Civitai resource metadata field when
any image is saved during that session.
"""

import gradio as gr
import html as _html
import json
import os
from pathlib import Path
import modules.scripts as scripts
from ch_lib import civitai, util, extra_resources
from modules_forge import presets as forge_presets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_resource_from_version_info(version_info: dict) -> dict:
    """Convert a Civitai version-info dict into a resource metadata entry."""
    model_type = version_info["model"]["type"].lower()
    if model_type in ["locon", "loha"]:
        model_type = "lycoris"
    resource = {
        "type": model_type,
        "modelVersionId": version_info["id"],
        "modelName": version_info["model"]["name"],
        "modelVersionName": version_info["name"],
    }
    if model_type in ("lora", "lycoris"):
        resource["weight"] = 1.0
    return resource


def _resources_html(tab_id: str = "") -> str:
    """Render the current pending list as an HTML snippet with per-item controls."""
    resources = extra_resources.get()
    if not resources:
        return "<p style='color:gray;margin:4px 0'>No extra resources in list.</p>"
    rows = []
    for i, r in enumerate(resources):
        weight_html = ""
        if r.get("type") in ("lora", "lycoris"):
            w = r.get("weight", 1.0)
            weight_html = (
                f"<label style='display:flex;align-items:center;gap:3px;"
                f"font-size:12px;flex-shrink:0'>"
                f"Weight"
                f"<input type='number' class='ch-extra-weight'"
                f" data-idx='{i}' data-tab='{tab_id}'"
                f" value='{w}' min='0' max='2' step='0.05'"
                f" style='width:60px;padding:1px 4px;font-size:12px;"
                f"border:1px solid var(--border-color-primary,#ccc);"
                f"border-radius:3px;"
                f"background:var(--input-background-fill,#fff);"
                f"color:var(--body-text-color,#000)'"
                f" title='LoRA weight'>"
                f"</label>"
            )
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;"
            f"border-bottom:1px solid var(--border-color-primary,#ddd)'>"
            f"<span style='flex:1'>"
            f"<b>{r.get('modelName', '?')}</b> &ndash; {r.get('modelVersionName', '?')} "
            f"<span style='color:var(--body-text-color-subdued,gray)'>"
            f"({r.get('type', '?')} &middot; version id: {r.get('modelVersionId', '?')})</span>"
            f"</span>"
            f"{weight_html}"
            f"<button class='ch-extra-rm' data-idx='{i}' data-tab='{tab_id}' "
            f"style='flex-shrink:0;cursor:pointer;"
            f"background:var(--button-cancel-background-fill,#dc3545);"
            f"color:var(--button-cancel-text-color,#fff);"
            f"border:none;border-radius:3px;padding:2px 8px;font-size:12px;line-height:1.5' "
            f"title='Remove'>&#x2715;</button>"
            f"</div>"
        )
    return f"<div style='margin:4px 0'>{''.join(rows)}</div>"


# ---------------------------------------------------------------------------
# Picker helpers
# ---------------------------------------------------------------------------

def _load_civitai_info_for_path(path) -> dict | None:
    """Load .civitai.info next to *path*, return parsed dict or None."""
    try:
        info_path = Path(path).with_suffix(".civitai.info")
        if not info_path.is_file():
            return None
        with open(info_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _read_sd_version_from_json(path) -> str | None:
    """Read sidecar [model_name].json and return normalized "sd version" value, if present."""
    try:
        json_path = Path(path).with_suffix(".json")
        if not json_path.is_file():
            return None
        with open(json_path, "r") as f:
            data = json.load(f)
        value = data.get("sd version")
        if value is None:
            return None
        return str(value).strip().lower()
    except Exception as e:
        util.printD(f"Error reading sd version from {path}: {e}")
        return None


def _default_picker_preset() -> str:
    """Best-effort default for the picker preset dropdown."""
    try:
        from modules import shared
        model = shared.sd_model
        if model is None:
            return forge_presets.PresetArch.sd.name
        if getattr(model, "is_sd1", False):
            return forge_presets.PresetArch.sd.name
        if getattr(model, "is_sdxl", False):
            return forge_presets.PresetArch.xl.name
        cfg_name = type(model.model_config).__name__.lower()
        if "flux2" in cfg_name:
            return forge_presets.PresetArch.klein.name
        if "flux" in cfg_name or "chroma" in cfg_name:
            return forge_presets.PresetArch.flux.name
        if "qwen" in cfg_name:
            return forge_presets.PresetArch.qwen.name
        if "lumina" in cfg_name:
            return forge_presets.PresetArch.lumina.name
        if "zimage" in cfg_name:
            return forge_presets.PresetArch.zit.name
        if "wan" in cfg_name:
            return forge_presets.PresetArch.wan.name
        if "anima" in cfg_name:
            return forge_presets.PresetArch.anima.name
        if "ernie" in cfg_name:
            return forge_presets.PresetArch.ernie.name
    except Exception:
        pass
    return forge_presets.PresetArch.sd.name


def _get_picker_items(resource_type: str, selected_preset: str | None = None) -> list[tuple[str, str, bool]]:
    """Return [(key, display_label, has_civitai_info)] for a resource type.

    For checkpoints and LoRAs, when selected_preset is provided,
    only items whose sidecar .json "sd version" equals that preset are included.
    """
    selected = (selected_preset or "").strip().lower()
    items = []
    if resource_type == "checkpoint":
        from modules import sd_models
        for title, info in sorted(sd_models.checkpoints_list.items(), key=lambda x: x[0].lower()):
            has_ci = Path(info.filename).with_suffix(".civitai.info").is_file()
            if selected:
                sd_version = _read_sd_version_from_json(info.filename)
                if sd_version != selected:
                    continue
            if has_ci:
                items.append((info.name, title, has_ci))
    elif resource_type == "lora":
        try:
            import networks as nets
            seen: set[str] = set()
            for name in sorted(nets.available_network_aliases.keys(), key=str.lower):
                net = nets.available_network_aliases[name]
                fn = str(net.filename)
                if fn in seen:
                    continue
                seen.add(fn)
                if selected:
                    sd_version = _read_sd_version_from_json(fn)
                    if sd_version != selected:
                        continue
                has_ci = Path(fn).with_suffix(".civitai.info").is_file()
                if has_ci:
                    items.append((name, name, has_ci))
        except Exception:
            pass
    elif resource_type == "upscaler":
        from modules import shared
        for up in sorted(shared.sd_upscalers, key=lambda x: x.name.lower()):
            if not up.data_path:
                continue
            has_ci = Path(up.data_path).with_suffix(".civitai.info").is_file()
            if has_ci:
                items.append((up.name, up.name, has_ci))
    elif resource_type == "embedding":
        from modules import shared
        try:
            from backend.args import dynamic_args
        except ModuleNotFoundError:
            dynamic_args = None
        embedding_dir = (
            (dynamic_args or {}).get("embedding_dir")
            or getattr(getattr(shared, "cmd_opts", None), "embeddings_dir", None)
        )
        if embedding_dir and os.path.isdir(embedding_dir):
            for dirpath, _, filenames in os.walk(embedding_dir, followlinks=True):
                for filename in sorted(filenames, key=str.lower):
                    fp = Path(dirpath) / filename
                    if fp.suffix.upper() not in (".BIN", ".PT", ".SAFETENSORS"):
                        continue
                    has_ci = fp.with_suffix(".civitai.info").is_file()
                    if has_ci:
                        items.append((fp.stem, fp.stem, has_ci))
    return items


def _picker_html(resource_type: str, tab_id: str, search: str = "", selected_preset: str | None = None) -> str:
    """Render a scrollable list of local models with '+' buttons for adding to the resource list."""
    search = (search or "").strip().lower()
    try:
        items = _get_picker_items(resource_type, selected_preset)
    except Exception as exc:
        return f"<p style='color:red;font-size:12px'>Error loading models: {exc}</p>"
    if search:
        items = [(k, l, h) for k, l, h in items if search in l.lower()]
    if not items:
        return "<p style='color:gray;font-size:12px;margin:4px 0'>No models found.</p>"
    rows = []
    for key, label, has_ci in items:
        mark = "\u2713" if has_ci else "\u2013"
        mark_color = "#2d8a4e" if has_ci else "var(--body-text-color-subdued,#999)"
        safe_key = _html.escape(key, quote=True)
        safe_label = _html.escape(label)
        safe_label_attr = _html.escape(label, quote=True)
        disabled_attr = "" if has_ci else "disabled"
        cursor = "pointer" if has_ci else "not-allowed"
        opacity = "1" if has_ci else "0.35"
        tip = "Add to extra resources" if has_ci else "Scan models in Civitai Helper tab first"
        rows.append(
            f"<div style='display:flex;align-items:center;gap:6px;padding:3px 2px;"
            f"border-bottom:1px solid var(--border-color-primary,#ddd)'>"
            f"<span style='color:{mark_color};flex-shrink:0;width:14px;text-align:center;"
            f"font-size:11px'>{mark}</span>"
            f"<span style='flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap' title='{safe_label_attr}'>{safe_label}</span>"
            f"<button class='ch-picker-add' {disabled_attr}"
            f" data-type='{resource_type}' data-key='{safe_key}' data-tab='{tab_id}'"
            f" style='flex-shrink:0;cursor:{cursor};opacity:{opacity};"
            f"background:var(--button-primary-background-fill,#1976d2);"
            f"color:var(--button-primary-text-color,#fff);"
            f"border:none;border-radius:3px;padding:1px 8px;font-size:14px;line-height:1.5'"
            f" title='{tip}'>+</button>"
            f"</div>"
        )
    content = "".join(rows)
    return (
        f"<div style='max-height:240px;overflow-y:auto'>{content}</div>"
        f"<p style='font-size:10px;color:var(--body-text-color-subdued,gray);margin:3px 0'>"
        f"{len(items)} model(s) &nbsp;|&nbsp; \u2713 = has Civitai info &nbsp;|&nbsp;"
        f" \u2013 = no info (scan first)</p>"
    )


def _build_resource_from_key(resource_type: str, key: str) -> dict | None:
    """Look up a local model by type + key and return a Civitai resource dict, or None."""
    ci = None
    if resource_type == "checkpoint":
        from modules import sd_models
        info = sd_models.get_closet_checkpoint_match(key)
        if info:
            ci = _load_civitai_info_for_path(info.filename)
    elif resource_type == "lora":
        try:
            import networks as nets
            net = nets.available_network_aliases.get(key)
            if net:
                ci = _load_civitai_info_for_path(net.filename)
        except Exception:
            pass
    elif resource_type == "upscaler":
        from modules import shared
        up = next((u for u in shared.sd_upscalers if u.name == key and u.data_path), None)
        if up:
            ci = _load_civitai_info_for_path(up.data_path)
    elif resource_type == "embedding":
        from modules import shared
        try:
            from backend.args import dynamic_args
        except ModuleNotFoundError:
            dynamic_args = None
        embedding_dir = (
            (dynamic_args or {}).get("embedding_dir")
            or getattr(getattr(shared, "cmd_opts", None), "embeddings_dir", None)
        )
        if embedding_dir:
            for dirpath, _, filenames in os.walk(embedding_dir, followlinks=True):
                for filename in filenames:
                    fp = Path(dirpath) / filename
                    if fp.stem == key and fp.suffix.upper() in (".BIN", ".PT", ".SAFETENSORS"):
                        ci = _load_civitai_info_for_path(fp)
                        break
                if ci is not None:
                    break
    if not ci:
        return None
    rtype = ci["model"]["type"].lower()
    if rtype in ("locon", "loha"):
        rtype = "lycoris"
    resource: dict = {
        "type": rtype,
        "modelVersionId": ci["id"],
        "modelName": ci["model"]["name"],
        "modelVersionName": ci["name"],
    }
    if rtype in ("lora", "lycoris"):
        resource["weight"] = 1.0
    return resource


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class Script(scripts.Script):

    def title(self):
        return "Civitai Helper"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab_id = "img2img" if is_img2img else "txt2img"

        with gr.Accordion(
            label="Civitai Helper",
            open=False,
            elem_id=f"ch_extra_accordion_{tab_id}",
        ):
            with gr.Row():
                enable_ckb = gr.Checkbox(
                    label="Enable",
                    value=extra_resources.get_enabled(),
                    elem_id=f"ch_extra_enable_{tab_id}",
                )
            gr.HTML(
                "<p style='margin:4px 0'>"
                "Add Civitai models by URL. "
                "These will be included in the <b>Civitai resources</b> metadata "
                "of every image generated this session. "
                "Use <b>Clear List</b> to remove them all."
                "</p>"
                "<p style='margin:6px 0 2px;font-size:0.9em;"
                "color:var(--body-text-color-subdued,gray)'>"
                "&#9432; Requires <b>Automatically add resource metadata to all generated images</b> "
                "to be enabled in "
                "<a class='ch_setting_link' href='#' style='color:var(--link-text-color,#1976d2)'>Settings &rsaquo; Civitai Helper</a>."
                "</p>"
            )

            with gr.Row():
                url_input = gr.Textbox(
                    label="Civitai Model URL",
                    placeholder="https://civitai.com/models/12345?modelVersionId=67890",
                    lines=1,
                    scale=4,
                    elem_id=f"ch_extra_url_{tab_id}",
                )
                fetch_btn = gr.Button(
                    value="Fetch & Add",
                    variant="primary",
                    scale=1,
                    elem_id=f"ch_extra_fetch_{tab_id}",
                )

            status_md = gr.Markdown(
                value="",
                elem_id=f"ch_extra_status_{tab_id}",
            )

            list_html = gr.HTML(
                value=_resources_html(tab_id),
                elem_id=f"ch_extra_list_{tab_id}",
            )

            with gr.Row():
                refresh_btn = gr.Button(
                    value="Refresh List",
                    elem_id=f"ch_extra_refresh_{tab_id}",
                )
                clear_btn = gr.Button(
                    value="Clear List",
                    variant="stop",
                    elem_id=f"ch_extra_clear_{tab_id}",
                )

            with gr.Accordion("Browse & Add Local Models", open=False, elem_id=f"ch_picker_accordion_{tab_id}"):
                picker_preset = gr.Dropdown(
                    label="Preset Filter (Checkpoint/LoRA)",
                    choices=forge_presets.PresetArch.choices(),
                    value=_default_picker_preset(),
                    elem_id=f"ch_picker_preset_{tab_id}",
                )
                with gr.Tabs(elem_id=f"ch_picker_tabs_{tab_id}"):
                    with gr.Tab("Checkpoints"):
                        ckpt_search = gr.Textbox(placeholder="Filter by name...", show_label=False, lines=1, elem_id=f"ch_picker_search_checkpoint_{tab_id}")
                        ckpt_html = gr.HTML(value=_picker_html("checkpoint", tab_id, selected_preset=picker_preset.value), elem_id=f"ch_picker_html_checkpoint_{tab_id}")
                        ckpt_reload = gr.Button("Reload", elem_id=f"ch_picker_reload_checkpoint_{tab_id}")

                    with gr.Tab("LoRA"):
                        lora_search = gr.Textbox(placeholder="Filter by name...", show_label=False, lines=1, elem_id=f"ch_picker_search_lora_{tab_id}")
                        lora_html = gr.HTML(value=_picker_html("lora", tab_id, selected_preset=picker_preset.value), elem_id=f"ch_picker_html_lora_{tab_id}")
                        lora_reload = gr.Button("Reload", elem_id=f"ch_picker_reload_lora_{tab_id}")

                    with gr.Tab("Upscalers"):
                        upscaler_search = gr.Textbox(placeholder="Filter by name...", show_label=False, lines=1, elem_id=f"ch_picker_search_upscaler_{tab_id}")
                        upscaler_html = gr.HTML(value=_picker_html("upscaler", tab_id), elem_id=f"ch_picker_html_upscaler_{tab_id}")
                        upscaler_reload = gr.Button("Reload", elem_id=f"ch_picker_reload_upscaler_{tab_id}")

                    with gr.Tab("Embeddings"):
                        embed_search = gr.Textbox(placeholder="Filter by name...", show_label=False, lines=1, elem_id=f"ch_picker_search_embedding_{tab_id}")
                        embed_html = gr.HTML(value=_picker_html("embedding", tab_id), elem_id=f"ch_picker_html_embedding_{tab_id}")
                        embed_reload = gr.Button("Reload", elem_id=f"ch_picker_reload_embedding_{tab_id}")

        # Hidden elements used by the inline JS remove buttons
        rm_idx_txtbox = gr.Textbox(
            value="", visible=False, elem_id=f"ch_extra_rm_idx_{tab_id}"
        )
        rm_hidden_btn = gr.Button(
            value="Remove", visible=False, elem_id=f"ch_extra_rm_btn_{tab_id}"
        )
        # Hidden elements used by the inline JS weight inputs
        weight_data_txtbox = gr.Textbox(
            value="", visible=False, elem_id=f"ch_extra_weight_data_{tab_id}"
        )
        weight_hidden_btn = gr.Button(
            value="Set Weight", visible=False, elem_id=f"ch_extra_weight_btn_{tab_id}"
        )
        # Hidden elements for the Browse & Add picker
        picker_add_key_txtbox = gr.Textbox(
            value="", visible=False, elem_id=f"ch_picker_add_key_{tab_id}"
        )
        picker_add_btn = gr.Button(
            value="Picker Add", visible=False, elem_id=f"ch_picker_add_btn_{tab_id}"
        )

        # ---- event handlers ------------------------------------------------

        def on_fetch(url):
            url = (url or "").strip()
            if not url:
                return _resources_html(tab_id), "", "Please enter a Civitai URL."

            result = civitai.get_model_id_from_url(url, include_model_ver=True)
            if not result:
                return _resources_html(tab_id), "", "Could not parse a model ID from this URL."

            model_id, model_version_id = result

            version_info = None
            if model_version_id:
                version_info = civitai.get_version_info_by_version_id(str(model_version_id))
            if not version_info:
                version_info = civitai.get_version_info_by_model_id(str(model_id))

            if not version_info:
                return _resources_html(tab_id), "", "Failed to fetch model info from Civitai."

            try:
                resource = _build_resource_from_version_info(version_info)
            except (KeyError, TypeError) as exc:
                return _resources_html(tab_id), "", f"Error parsing model info: {exc}"

            extra_resources.add(resource)
            return (
                _resources_html(tab_id),
                "",
                f"Added: **{resource['modelName']}** &ndash; {resource['modelVersionName']}",
            )

        def on_remove_inline(idx_str):
            try:
                extra_resources.remove(int(idx_str))
            except (ValueError, IndexError):
                pass
            return _resources_html(tab_id), "", ""

        def on_clear():
            extra_resources.clear()
            return _resources_html(tab_id), "List cleared."

        fetch_btn.click(
            on_fetch,
            inputs=[url_input],
            outputs=[list_html, url_input, status_md],
        )

        rm_hidden_btn.click(
            on_remove_inline,
            inputs=[rm_idx_txtbox],
            outputs=[list_html, url_input, status_md],
        )

        def on_set_weight(data_str):
            try:
                idx_str, weight_str = data_str.split(":", 1)
                extra_resources.set_weight(int(idx_str), float(weight_str))
            except (ValueError, IndexError):
                pass

        weight_hidden_btn.click(
            on_set_weight,
            inputs=[weight_data_txtbox],
            outputs=[],
        )

        refresh_btn.click(
            fn=lambda: _resources_html(tab_id),
            outputs=[list_html],
        )

        clear_btn.click(
            on_clear,
            outputs=[list_html, status_md],
        )

        enable_ckb.change(
            extra_resources.set_enabled,
            inputs=[enable_ckb],
        )

        # Picker: search filter live-update + reload
        picker_preset.change(
            fn=lambda p, s1, s2: (
                _picker_html("checkpoint", tab_id, s1, p),
                _picker_html("lora", tab_id, s2, p),
            ),
            inputs=[picker_preset, ckpt_search, lora_search],
            outputs=[ckpt_html, lora_html],
        )
        ckpt_search.change(fn=lambda s, p: _picker_html("checkpoint", tab_id, s, p), inputs=[ckpt_search, picker_preset], outputs=[ckpt_html])
        ckpt_reload.click(fn=lambda p: _picker_html("checkpoint", tab_id, selected_preset=p), inputs=[picker_preset], outputs=[ckpt_html])
        lora_search.change(fn=lambda s, p: _picker_html("lora", tab_id, s, p), inputs=[lora_search, picker_preset], outputs=[lora_html])
        lora_reload.click(fn=lambda p: _picker_html("lora", tab_id, selected_preset=p), inputs=[picker_preset], outputs=[lora_html])
        upscaler_search.change(fn=lambda s: _picker_html("upscaler", tab_id, s), inputs=[upscaler_search], outputs=[upscaler_html])
        upscaler_reload.click(fn=lambda: _picker_html("upscaler", tab_id), outputs=[upscaler_html])
        embed_search.change(fn=lambda s: _picker_html("embedding", tab_id, s), inputs=[embed_search], outputs=[embed_html])
        embed_reload.click(fn=lambda: _picker_html("embedding", tab_id), outputs=[embed_html])

        # Picker: add button (triggered by JS on .ch-picker-add click)
        def on_picker_add(key_json):
            try:
                data = json.loads(key_json)
                resource_type = data["type"]
                key = data["key"]
            except Exception:
                return _resources_html(tab_id), "", "Invalid picker selection."
            resource = _build_resource_from_key(resource_type, key)
            if resource:
                existing_ids = {r.get("modelVersionId") for r in extra_resources.get()}
                if resource.get("modelVersionId") in existing_ids:
                    return _resources_html(tab_id), "", f"Already in list: **{resource.get('modelName', key)}**"
                extra_resources.add(resource)
                return (
                    _resources_html(tab_id),
                    "",
                    f"Added: **{resource['modelName']}** \u2013 {resource['modelVersionName']}",
                )
            return _resources_html(tab_id), "", f"Could not load Civitai info for **{key}** ({resource_type})."

        picker_add_btn.click(
            on_picker_add,
            inputs=[picker_add_key_txtbox],
            outputs=[list_html, picker_add_key_txtbox, status_md],
        )

        # No components need to be passed through to process()
        return []

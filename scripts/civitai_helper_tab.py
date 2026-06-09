""" -*- coding: UTF-8 -*-
Adds a "Civitai Helper" accordion to the txt2img and img2img tabs.

Users can paste Civitai model URLs, fetch metadata, and build a temporary list
of resources that will be appended to the Civitai resource metadata field when
any image is saved during that session.
"""

import gradio as gr
import modules.scripts as scripts
from ch_lib import civitai, util, extra_resources


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

        # No components need to be passed through to process()
        return []

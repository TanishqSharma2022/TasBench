#!/usr/bin/env python3
"""
Generate a comprehensive, beautiful side-by-side HTML comparison dashboard
for all 11 TSR models on en.jpg and hi.jpg.
"""
import os
import json
import base64
import re

RESULTS_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/inference_results"
IMAGES_DIR = "/home/tanish/scratch/tanishq/table-understanding/repos/synthetic-table-generation/rendered_tables"
OUTPUT_HTML = os.path.join(RESULTS_DIR, "comparison_dashboard.html")

MODELS = [
    {
        "id": "pixtab",
        "name": "PIX-TAB",
        "paradigm": "Autoregressive (PAPP-OTSL + BBox Tokens)",
        "venue": "CVPR 2026",
        "checkpoint": "epoch_25.pth (full-pubtabnet-500k-resume)",
        "has_bbox": True,
    },
    {
        "id": "mtltabnet",
        "name": "MTL-TabNet",
        "paradigm": "Autoregressive Multi-Task Decoder",
        "venue": "ICDAR 2021",
        "checkpoint": "epoch_19.pth (PubTabNet)",
        "has_bbox": True,
    },
    {
        "id": "unitable",
        "name": "UniTable",
        "paradigm": "Autoregressive Unified Transformer (Structure + BBox Head)",
        "venue": "NeurIPS 2024W",
        "checkpoint": "unitable_large_structure.pt + unitable_large_bbox.pt",
        "has_bbox": True,
    },
    {
        "id": "tableformer",
        "name": "TableFormer V2",
        "paradigm": "OTSL Structure + Cell BBox Attention Head",
        "venue": "Docling 2025 / IBM",
        "checkpoint": "docling-project/TableFormerV2 (HF)",
        "has_bbox": True,
    },
    {
        "id": "slanet",
        "name": "SLANet-plus",
        "paradigm": "Autoregressive Structure-Location Alignment",
        "venue": "PaddleOCR PP-StructureV3",
        "checkpoint": "slanet_plus (rapid-table)",
        "has_bbox": True,
    },
    {
        "id": "tatr",
        "name": "TATR v1.1-All",
        "paradigm": "Detection (DETR Row/Col/Span Queries)",
        "venue": "CVPR 2023 / Microsoft",
        "checkpoint": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "has_bbox": True,
    },
    {
        "id": "lore",
        "name": "LORE-TSR",
        "paradigm": "Detection + Logical Location Regression",
        "venue": "AAAI 2023 / Alibaba",
        "checkpoint": "lineless_table_rec (RapidAI)",
        "has_bbox": True,
    },
    {
        "id": "cycle_centernet",
        "name": "Cycle-CenterNet",
        "paradigm": "Geometric Detection (Corner/Center Keypoints)",
        "venue": "ICCV 2021 / Alibaba",
        "checkpoint": "wired_table_rec (RapidAI)",
        "has_bbox": True,
    },
    {
        "id": "smoldocling",
        "name": "SmolDocling-256M",
        "paradigm": "Autoregressive Document VLM",
        "venue": "ICCV 2025 / IBM-HF",
        "checkpoint": "ds4sd/SmolDocling-256M-preview",
        "has_bbox": True,
    },
    {
        "id": "sprint",
        "name": "SPRINT",
        "paradigm": "Autoregressive Sequence-to-Structure",
        "venue": "ICDAR 2023",
        "checkpoint": "SPRINT checkpoint (PubTabNet)",
        "has_bbox": False,
    },
    {
        "id": "tflop",
        "name": "TFLOP",
        "paradigm": "Autoregressive Sequence Generator",
        "venue": "Springer 2023",
        "checkpoint": "tflop_pubtabnet.pth",
        "has_bbox": False,
    },
]

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def count_table_elements(html):
    rows = len(re.findall(r"<tr", html, re.IGNORECASE))
    cells = len(re.findall(r"<td|<th", html, re.IGNORECASE))
    return rows, cells

def generate_dashboard():
    splits_data = {}
    for split in ["en", "hi"]:
        img_name = f"{split}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_name)
        img_b64 = img_to_b64(img_path)
        
        models_info = []
        for m in MODELS:
            mid = m["id"]
            raw_path = os.path.join(RESULTS_DIR, split, f"{mid}_raw.json")
            html_path = os.path.join(RESULTS_DIR, split, f"{mid}.html")
            bbox_img_path = os.path.join(RESULTS_DIR, split, f"{mid}_bboxes.png")
            
            raw_data = {}
            if os.path.exists(raw_path):
                try:
                    with open(raw_path) as f:
                        raw_data = json.load(f)
                except Exception as e:
                    raw_data = {"error": str(e)}

            html_content = ""
            if os.path.exists(html_path):
                with open(html_path) as f:
                    html_content = f.read().strip()
            
            rows_cnt, cells_cnt = count_table_elements(html_content)
            
            bbox_b64 = img_to_b64(bbox_img_path) if os.path.exists(bbox_img_path) else ""
            
            models_info.append({
                "meta": m,
                "raw": raw_data,
                "html": html_content,
                "rows": rows_cnt,
                "cells": cells_cnt,
                "bbox_b64": bbox_b64,
                "bbox_path": f"{split}/{mid}_bboxes.png" if os.path.exists(bbox_img_path) else "",
                "latency": raw_data.get("elapse_sec", None),
                "num_bboxes": len(raw_data.get("cell_bboxes", [])) if "cell_bboxes" in raw_data and raw_data["cell_bboxes"] else (len(raw_data.get("bboxes", [])) if "bboxes" in raw_data and raw_data["bboxes"] else 0)
            })
            
        splits_data[split] = {
            "img_b64": img_b64,
            "img_path": f"../synthetic-table-generation/rendered_tables/{img_name}",
            "models": models_info
        }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Table Structure Recognition (TSR) - 11 Models Benchmark Dashboard</title>
<style>
  :root {{
    --primary: #1e40af;
    --primary-light: #3b82f6;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --text-muted: #64748b;
    --header-bg: #1e293b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding-bottom: 60px;
  }}

  header {{
    background: var(--header-bg);
    color: white;
    padding: 24px 32px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }}
  header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; }}
  header p {{ font-size: 14px; color: #94a3b8; }}

  .container {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 24px 20px;
  }}

  /* Tab Navigation */
  .tab-nav {{
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 8px;
  }}
  .tab-btn {{
    padding: 10px 22px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 8px 8px 0 0;
    background: #e2e8f0;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
  }}
  .tab-btn.active {{
    background: var(--primary);
    color: white;
  }}
  .tab-btn:hover:not(.active) {{
    background: #cbd5e1;
    color: var(--text);
  }}

  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Reference Image Banner */
  .reference-card {{
    background: white;
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
  }}
  .reference-img-box {{
    flex: 0 0 auto;
    max-width: 480px;
  }}
  .reference-img-box img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }}
  .reference-meta {{
    flex: 1 1 300px;
  }}
  .reference-meta h2 {{ font-size: 20px; color: var(--primary); margin-bottom: 8px; }}
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 9999px;
    background: #dbeafe;
    color: #1e40af;
    margin-bottom: 12px;
  }}

  /* Summary Table */
  .summary-table-wrap {{
    background: white;
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 16px 20px;
    margin-bottom: 30px;
    overflow-x: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .summary-table th, .summary-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }}
  .summary-table th {{
    background: #f1f5f9;
    font-weight: 600;
    color: var(--text);
  }}
  .summary-table tr:hover {{ background-color: #f8fafc; }}

  /* Model Card Grid */
  .model-card {{
    background: white;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 32px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    overflow: hidden;
  }}
  .model-card-header {{
    background: #f8fafc;
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .model-title {{
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .paradigm-pill {{
    font-size: 12px;
    padding: 3px 8px;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 6px;
    font-weight: 500;
  }}
  .stats-badges {{
    display: flex;
    gap: 8px;
    font-size: 12px;
  }}
  .stat-badge {{
    padding: 4px 10px;
    border-radius: 6px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    font-weight: 600;
  }}

  .model-card-body {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 20px;
  }}
  @media (max-width: 1024px) {{
    .model-card-body {{ grid-template-columns: 1fr; }}
  }}

  .pane-header {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  /* Bounding Box Image Pane */
  .bbox-pane {{
    display: flex;
    flex-direction: column;
  }}
  .bbox-img-wrap {{
    background: #0f172a;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
    min-height: 300px;
  }}
  .bbox-img-wrap img {{
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
    border-radius: 4px;
  }}
  .no-bbox-notice {{
    color: #94a3b8;
    text-align: center;
    padding: 40px 20px;
  }}

  /* Rendered HTML Table Pane */
  .html-pane {{
    display: flex;
    flex-direction: column;
  }}
  .table-render-wrap {{
    flex: 1;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    max-height: 500px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
  }}

  /* High-Visibility Table Styles */
  .table-render-wrap table {{
    border-collapse: collapse !important;
    width: 100% !important;
    font-family: inherit;
    font-size: 12px;
    margin: 0 auto;
  }}
  .table-render-wrap th, 
  .table-render-wrap td {{
    border: 1.5px solid #1e293b !important;
    padding: 6px 10px !important;
    text-align: center !important;
    min-width: 28px;
    min-height: 22px;
    background-color: #ffffff;
  }}
  .table-render-wrap th,
  .table-render-wrap thead td {{
    background-color: #e2e8f0 !important;
    font-weight: 700;
    color: #0f172a;
  }}
  .table-render-wrap tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}

  /* Collapsible Raw Code */
  details {{
    margin-top: 14px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }}
  summary {{
    font-size: 12px;
    font-weight: 600;
    color: var(--primary-light);
    cursor: pointer;
    user-select: none;
  }}
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 12px;
    border-radius: 6px;
    font-size: 11px;
    overflow-x: auto;
    margin-top: 8px;
    max-height: 200px;
  }}
</style>
</head>
<body>

<header>
  <div class="container" style="padding: 0;">
    <h1>Table Structure Recognition (TSR) — 11 SOTA Models Comparative Benchmark</h1>
    <p>Comprehensive side-by-side evaluation of Autoregressive, Detection/Regression, OTSL, and Document VLM Paradigms on English & Hindi Tables.</p>
  </div>
</header>

<div class="container">
  <div class="tab-nav">
    <button class="tab-btn active" onclick="switchTab('en')">English Table (en.jpg - 796x610)</button>
    <button class="tab-btn" onclick="switchTab('hi')">Hindi Table (hi.jpg - 808x610)</button>
  </div>
"""

    for split in ["en", "hi"]:
        s_data = splits_data[split]
        active_cls = " active" if split == "en" else ""
        lang_title = "English Complex Financial Table (en.jpg)" if split == "en" else "Hindi Complex Synthetic Table (hi.jpg)"
        
        html += f"""
  <div id="tab-{split}" class="tab-content{active_cls}">
    <!-- Reference Image Header -->
    <div class="reference-card">
      <div class="reference-img-box">
        <img src="{s_data['img_b64']}" alt="Original {split}.jpg">
      </div>
      <div class="reference-meta">
        <span class="badge">Ground Truth Input Image</span>
        <h2>{lang_title}</h2>
        <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 12px;">
          File: <code>synthetic-table-generation/rendered_tables/{split}.jpg</code><br>
          Dimensions: {'796 x 610 px' if split == 'en' else '808 x 610 px'} | Borderless / Lineless Table Challenge
        </p>
        <p style="font-size: 13px;">
          This evaluation tests each model's resilience to row/col spanning, multi-header hierarchy, and zero-border cell adjacency.
        </p>
      </div>
    </div>

    <!-- Summary Metrics Table -->
    <div class="summary-table-wrap">
      <h3 style="font-size: 15px; margin-bottom: 10px; font-weight: 700;">Model Prediction Summary ({split.upper()})</h3>
      <table class="summary-table">
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Paradigm</th>
            <th>Venue</th>
            <th>Predicted Rows</th>
            <th>Predicted Cells</th>
            <th>Cell BBoxes</th>
            <th>Inference Latency</th>
          </tr>
        </thead>
        <tbody>
"""
        for m_res in s_data["models"]:
            meta = m_res["meta"]
            lat_str = f"{m_res['latency']:.2f}s" if m_res['latency'] else "N/A"
            bbox_status = f"<span style='color: #16a34a; font-weight: 600;'>{m_res['num_bboxes']} bboxes</span>" if meta["has_bbox"] and m_res["num_bboxes"] > 0 else "<span style='color: #64748b;'>None (Structure Only)</span>"
            html += f"""
          <tr>
            <td><strong>{meta['name']}</strong></td>
            <td><span class="paradigm-pill">{meta['paradigm'].split('(')[0].strip()}</span></td>
            <td>{meta['venue']}</td>
            <td><strong>{m_res['rows']}</strong></td>
            <td><strong>{m_res['cells']}</strong></td>
            <td>{bbox_status}</td>
            <td>{lat_str}</td>
          </tr>
"""
        html += """
        </tbody>
      </table>
    </div>

    <!-- Side-by-Side Model Cards -->
"""
        for m_res in s_data["models"]:
            meta = m_res["meta"]
            lat_str = f"{m_res['latency']:.2f}s" if m_res['latency'] else "N/A"
            escaped_html = m_res["html"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # Left pane: BBoxes or Original
            if m_res["bbox_b64"]:
                left_pane = f"""
        <div class="bbox-pane">
          <div class="pane-header">
            <span>Predicted Cell Bounding Boxes ({m_res['num_bboxes']} boxes)</span>
            <span style="font-size: 11px; color: #16a34a; font-weight: 600;">Overlay Enabled</span>
          </div>
          <div class="bbox-img-wrap">
            <img src="{m_res['bbox_b64']}" alt="{meta['name']} BBoxes">
          </div>
        </div>
"""
            else:
                left_pane = f"""
        <div class="bbox-pane">
          <div class="pane-header">
            <span>Input Table Image</span>
            <span style="font-size: 11px; color: #64748b;">Structure Only Head</span>
          </div>
          <div class="bbox-img-wrap" style="flex-direction: column;">
            <img src="{s_data['img_b64']}" alt="Original Input">
            <div class="no-bbox-notice">This model architecture outputs structural HTML/OTSL tokens without spatial cell bounding boxes.</div>
          </div>
        </div>
"""

            html += f"""
    <div class="model-card" id="{split}_{meta['id']}">
      <div class="model-card-header">
        <div class="model-title">
          <span>{meta['name']}</span>
          <span class="paradigm-pill">{meta['paradigm']}</span>
        </div>
        <div class="stats-badges">
          <div class="stat-badge">Venue: {meta['venue']}</div>
          <div class="stat-badge">Rows: {m_res['rows']}</div>
          <div class="stat-badge">Cells: {m_res['cells']}</div>
          <div class="stat-badge">Latency: {lat_str}</div>
        </div>
      </div>
      <div class="model-card-body">
        {left_pane}
        <div class="html-pane">
          <div class="pane-header">
            <span>Predicted HTML Structure (Rendered with 1.5px Borders)</span>
            <span style="font-size: 11px; color: var(--text-muted);">{m_res['rows']} Rows &bull; {m_res['cells']} Cells</span>
          </div>
          <div class="table-render-wrap">
            {m_res['html']}
          </div>
        </div>
      </div>
      <div style="padding: 0 20px 16px 20px;">
        <details>
          <summary>View Raw HTML Token Stream & Checkpoint Details</summary>
          <p style="margin-top: 8px; font-size: 12px; color: var(--text-muted);"><strong>Checkpoint / Origin:</strong> {meta['checkpoint']}</p>
          <pre><code>{escaped_html}</code></pre>
        </details>
      </div>
    </div>
"""
        html += """
  </div>
"""

    html += """
</div>

<script>
function switchTab(split) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  
  if (split === 'en') {
    document.querySelectorAll('.tab-btn')[0].classList.add('active');
    document.getElementById('tab-en').classList.add('active');
  } else {
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
    document.getElementById('tab-hi').classList.add('active');
  }
}
</script>

</body>
</html>
"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Comparison dashboard successfully written to: {OUTPUT_HTML}")
    print(f"Dashboard file size: {os.path.getsize(OUTPUT_HTML) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    generate_dashboard()

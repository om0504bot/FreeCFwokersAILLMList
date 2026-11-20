#!/usr/bin/env python3
"""
Scrape Cloudflare Workers AI models and generate comparison table.
Runs daily via GitHub Actions, outputs to GitHub Pages.
"""

import re
import json
import requests
from datetime import datetime, timezone
from urllib.parse import urljoin
from pathlib import Path

# --- Constants ---
LLMS_TXT_URL = "https://developers.cloudflare.com/workers-ai/llms.txt"
PRICING_URL = "https://developers.cloudflare.com/workers-ai/platform/pricing/"
LIMITS_URL = "https://developers.cloudflare.com/workers-ai/platform/limits/"
BASE_URL = "https://developers.cloudflare.com"

# Models that REQUIRE paid billing (from pricing page)
PAID_MODELS = {
    "@cf/moonshotai/kimi-k2.6",
    "@cf/moonshotai/kimi-k2.7-code",
    "@cf/zai-org/glm-5.2",
    "@cf/zai-org/glm-5.3",
    "@cf/zai-org/glm-5.3-flash",
    "@cf/deepseek-ai/deepseek-v4-flash-0731",
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
    "@cf/deepseek-ai/deepseek-v4-pro-0813",
    "@cf/thinkingmachines/inkling-2-small",
}

# Country inference from org name / model name
COUNTRY_MAP = {
    "meta": "USA", "google": "USA", "microsoft": "USA", "mistralai": "France",
    "deepseek": "China", "zai-org": "China", "moonshotai": "China",
    "baai": "China", "ai4bharat": "India", "aisingapore": "Singapore",
    "bytedance": "China", "black-forest-labs": "Germany",
    "daily-co": "USA", "deepgram": "USA", "deepl": "Germany",
    "nvidia": "USA", "thinkingmachines": "USA",
    "cloudflare": "USA", "runwayml": "USA", "stabilityai": "UK",
    "sentence-transformers": "Germany", "jina-ai": "Germany",
    "nomic-ai": "USA", "mixedbread-ai": "Germany",
    "infgrad": "China", "upstage": "South Korea",
    "ibm": "USA", "allenai": "USA", "ml6team": "Belgium",
}

TASK_TYPE_MAP = {
    "text-generation": "文字生成",
    "text-embeddings": "文字嵌入",
    "image-generation": "圖片生成",
    "image-classification": "圖片分類",
    "object-detection": "物件偵測",
    "translation": "翻譯",
    "summarization": "摘要",
    "text-classification": "文字分類",
    "automatic-speech-recognition": "語音辨識",
    "text-to-speech": "語音合成",
    "image-to-text": "圖片轉文字",
}


def infer_country(model_id: str) -> str:
    """Infer country from model ID organization."""
    parts = model_id.split("/")
    if len(parts) >= 2:
        org = parts[1].lower()
        for key, country in COUNTRY_MAP.items():
            if key in org:
                return country
    return "未知"


def infer_task_type(model_id: str, model_name: str) -> str:
    """Infer task type from model ID/name."""
    model_lower = f"{model_id} {model_name}".lower()
    
    if any(k in model_lower for k in ["embed", "bge", "e5", "gte", "nomic", "jina", "mixedbread"]):
        return "text-embeddings"
    if any(k in model_lower for k in ["flux", "stable-diffusion", "sdxl", "image-gen", "text-to-image"]):
        return "image-generation"
    if any(k in model_lower for k in ["whisper", "nova", "aura", "phonellm", "speech", "tts", "stt"]):
        return "automatic-speech-recognition" if "stt" in model_lower or "recognition" in model_lower else "text-to-speech"
    if any(k in model_lower for k in ["vision", "llava", "qwen-vl", "visual"]):
        return "image-to-text"
    if any(k in model_lower for k in ["translate", "indictrans", "nllb", "m2m"]):
        return "translation"
    if any(k in model_lower for k in ["rerank", "reranker"]):
        return "text-classification"
    if any(k in model_lower for k in ["classif", "detect", "yolo", "object"]):
        return "object-detection"
    if any(k in model_lower for k in ["summar"]):
        return "summarization"
    return "text-generation"


def fetch_llms_txt() -> list[dict]:
    """Parse llms.txt to get all model IDs and detail URLs."""
    resp = requests.get(LLMS_TXT_URL, timeout=30)
    resp.raise_for_status()
    
    models = []
    # Pattern: - [@cf/org/model-name](https://developers.cloudflare.com/workers-ai/models/model-name/index.md)
    pattern = r'-\s*\[(@cf/[^\]]+)\]\((https://developers\.cloudflare\.com/workers-ai/models/[^\)]+)\)'
    
    for match in re.finditer(pattern, resp.text):
        model_id = match.group(1)
        detail_url = match.group(2)
        # Extract model slug from URL for display name
        slug = detail_url.split("/models/")[1].split("/")[0]
        display_name = slug.replace("-", " ").title()
        
        models.append({
            "id": model_id,
            "detail_url": detail_url,
            "display_name": display_name,
            "slug": slug,
        })
    
    return models


def fetch_pricing_table() -> dict[str, dict]:
    """Parse pricing page for per-model neuron costs."""
    resp = requests.get(PRICING_URL, timeout=30)
    resp.raise_for_status()
    
    pricing = {}
    # Find the LLM model pricing table
    # Pattern: | @cf/org/model | $X per M input tokens $Y per M output tokens | N neurons per M input tokens M neurons per M output tokens |
    table_pattern = r'\|\s*(@cf/[^|]+)\s*\|\s*\$([\d.]+)\s*per\s*M\s*input\s*tokens\s*\$([\d.]+)\s*per\s*M\s*output\s*tokens\s*\|\s*([\d,]+)\s*neurons\s*per\s*M\s*input\s*tokens\s*([\d,]+)\s*neurons\s*per\s*M\s*output\s*tokens'
    
    for match in re.finditer(table_pattern, resp.text):
        model_id = match.group(1).strip()
        input_token_price = float(match.group(2))
        output_token_price = float(match.group(3))
        input_neurons = int(match.group(4).replace(",", ""))
        output_neurons = int(match.group(5).replace(",", ""))
        
        pricing[model_id] = {
            "input_token_price": input_token_price,
            "output_token_price": output_token_price,
            "input_neurons_per_m": input_neurons,
            "output_neurons_per_m": output_neurons,
        }
    
    return pricing


def fetch_rate_limits() -> dict[str, dict]:
    """Parse limits page for rate limits."""
    resp = requests.get(LIMITS_URL, timeout=30)
    resp.raise_for_status()
    
    limits = {}
    
    # Default limits by task type
    task_limits = {
        "text-generation": 300,
        "text-embeddings": 3000,
        "image-generation": 720,
        "image-classification": 3000,
        "object-detection": 3000,
        "translation": 720,
        "summarization": 1500,
        "text-classification": 2000,
        "automatic-speech-recognition": 720,
        "text-to-speech": 720,
        "image-to-text": 720,
    }
    
    # Paid model limits
    paid_limits = {
        "standard": 20,
        "ai_gateway": 50,
    }
    
    # Check for model-specific limits (bge-large-en-v1.5, stable-diffusion-v1-5-img2img)
    if "@cf/baai/bge-large-en-v1.5" in resp.text:
        limits["@cf/baai/bge-large-en-v1.5"] = {"rpm": 1500, "note": "較低限制"}
    if "@cf/runwayml/stable-diffusion-v1-5-img2img" in resp.text:
        limits["@cf/runwayml/stable-diffusion-v1-5-img2img"] = {"rpm": 1500, "note": "較低限制"}
    
    return {"task_defaults": task_limits, "paid_limits": paid_limits, "model_specific": limits}


def check_deprecated(models: list[dict]) -> dict[str, str]:
    """Check individual model pages for deprecation notices."""
    deprecated = {}
    # Known deprecated/preview models from documentation patterns
    known_deprecated = {
        "@cf/cloudflare/clairic": "研究預覽版",
        "@cf/cloudflare/clairic-flash": "早期研究預覽版",
        "@cf/cloudflare/clef": "預覽版",
        "@cf/cloudflare/clef-flash": "預覽版",
        "@cf/cloudflare/clef-omni": "預覽版",
        "@cf/daily-co/phonellm-mini-alpha-2": "Alpha 版本",
        "@cf/deepseek-ai/deepseek-v4-flash-0731": "已由 v4.1 取代",
    }
    
    for model in models:
        model_id = model["id"]
        if model_id in known_deprecated:
            deprecated[model_id] = known_deprecated[model_id]
    
    return deprecated


def main():
    print("Fetching model list from llms.txt...")
    models = fetch_llms_txt()
    print(f"Found {len(models)} models")
    
    print("Fetching pricing data...")
    pricing = fetch_pricing_table()
    print(f"Found pricing for {len(pricing)} models")
    
    print("Fetching rate limits...")
    limits = fetch_rate_limits()
    
    print("Checking deprecation status...")
    deprecated = check_deprecated(models)
    
    # Build enriched model data
    enriched = []
    free_count = 0
    paid_count = 0
    
    for model in models:
        model_id = model["id"]
        is_paid = model_id in PAID_MODELS
        is_free = not is_paid
        
        if is_free:
            free_count += 1
        else:
            paid_count += 1
        
        task_type = infer_task_type(model_id, model["display_name"])
        task_type_zh = TASK_TYPE_MAP.get(task_type, task_type)

        # Add tool calling support field (True for text-generation models that support function calling)
        # Text-generation models typically support tool calling; others may not
        tool_calling = task_type == "text-generation"

        # Get pricing info
        price_info = pricing.get(model_id, {})

        # Get rate limit
        model_limits = limits["model_specific"].get(model_id, {})
        if model_limits:
            rpm = model_limits.get("rpm", 0)
            rpm_note = model_limits.get("note", "")
        elif is_paid:
            rpm = limits["paid_limits"]["standard"]
            rpm_note = "付費模型標準限制"
        else:
            rpm = limits["task_defaults"].get(task_type, 0)
            rpm_note = ""

        # Free tier neurons per day
        free_neurons_day = 10000

        # Estimate free requests per day based on typical usage
        # Assume average 1K input + 1K output tokens per request
        avg_neurons_per_req = 0
        if price_info:
            avg_neurons_per_req = (price_info.get("input_neurons_per_m", 0) + price_info.get("output_neurons_per_m", 0)) / 1000
        est_free_requests = int(free_neurons_day / avg_neurons_per_req) if avg_neurons_per_req > 0 else "N/A"

        enriched.append({
            "id": model_id,
            "display_name": model["display_name"],
            "detail_url": model["detail_url"],
            "is_free": is_free,
            "is_paid": is_paid,
            "country": infer_country(model_id),
            "task_type": task_type,
            "task_type_zh": task_type_zh,
            "tool_calling": "✅ 支援" if tool_calling else "❌ 不支援",
            "deprecated": deprecated.get(model_id, ""),
            "pricing": price_info,
            "rpm": rpm,
            "rpm_note": rpm_note,
            "free_neurons_day": free_neurons_day,
            "est_free_requests": est_free_requests,
        })
    
    # Sort: free models first, then by task type, then by ID
    enriched.sort(key=lambda x: (not x["is_free"], x["task_type"], x["id"]))
    
    # Generate outputs
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    # JSON for programmatic use
    with open(output_dir / "models.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_models": len(enriched),
            "free_models": free_count,
            "paid_models": paid_count,
            "models": enriched,
        }, f, ensure_ascii=False, indent=2)
    
    # HTML for GitHub Pages
    generate_html(enriched, output_dir / "index.html", free_count, paid_count)
    
    print(f"Done! Free: {free_count}, Paid: {paid_count}")
    print(f"Output written to {output_dir}/")


def generate_html(models: list[dict], output_path: Path, free_count: int, paid_count: int):
    """Generate the comparison table HTML."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    # Separate free and paid
    free_models = [m for m in models if m["is_free"]]
    paid_models = [m for m in models if m["is_paid"]]
    
    def render_table(model_list: list[dict], title: str) -> str:
        rows = []
        for m in model_list:
            price = m["pricing"]
            deprecated_badge = ""
            if m["deprecated"]:
                deprecated_badge = f'<span class="badge deprecated" title="{m["deprecated"]}">⚠ {m["deprecated"]}</span>'
            
            # Pricing display
            if price:
                price_str = f"${price['input_token_price']:.3f}/M in<br>${price['output_token_price']:.3f}/M out"
                neuron_str = f"{price['input_neurons_per_m']:,} / {price['output_neurons_per_m']:,} neurons/M"
            else:
                price_str = "—"
                neuron_str = "—"
            
            # Free tier estimate
            if isinstance(m["est_free_requests"], int):
                free_est = f"~{m['est_free_requests']:,} 次/天"
            else:
                free_est = "—"
            
            rows.append(f"""
            <tr class="{'free' if m['is_free'] else 'paid'}">
                <td><code class="model-id" title="點擊複製">{m['id']}</code></td>
                <td><a href="{m['detail_url']}" target="_blank" rel="noopener">{m['display_name']}</a></td>
                <td>{m['task_type_zh']}</td>
                <td>{m['country']}</td>
                <td>{price_str}</td>
                <td>{neuron_str}</td>
                <td>{free_est}</td>
                <td>{m['tool_calling']}</td>
                <td>{m['rpm']:,} req/min {m['rpm_note']}</td>
                <td>{deprecated_badge}</td>
            </tr>""")
        
        return f"""
        <h2>{title} ({len(model_list)})</h2>
        <table>
            <thead>
                <tr>
                    <th>Model ID <span class="copy-hint">(點擊複製)</span></th>
                    <th>模型名稱 (連結到詳情)</th>
                    <th>任務類型</th>
                    <th>國家/組織</th>
                    <th>Token 價格</th>
                    <th>Neurons 價格</th>
                    <th>免費額度估算</th>
                    <th>工具支援</th>
                    <th>速率限制</th>
                    <th>狀態</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloudflare Workers AI 免費模型比較表</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
            line-height: 1.6; color: #1a1a2e; background: #f8f9fa;
            margin: 0; padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 30px; }}
        h1 {{ color: #f38020; margin-bottom: 8px; font-size: 1.8rem; }}
        .meta {{ color: #666; margin-bottom: 24px; font-size: 0.9rem; }}
        .stats {{ display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }}
        .stat-card {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px 24px; min-width: 180px; }}
        .stat-card.free {{ background: #f0fdf4; border-color: #bbf7d0; }}
        .stat-card.paid {{ background: #fef2f2; border-color: #fecaca; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #f38020; }}
        .stat-card.free .stat-value {{ color: #16a34a; }}
        .stat-card.paid .stat-value {{ color: #dc2626; }}
        .stat-label {{ font-size: 0.85rem; color: #666; margin-top: 4px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; color: #374151; position: sticky; top: 0; }}
        tr:hover {{ background: #f9fafb; }}
        tr.free {{ background: #f0fdf4; }}
        tr.free:hover {{ background: #dcfce7; }}
        tr.paid {{ background: #fef2f2; }}
        tr.paid:hover {{ background: #fee2e2; }}
        
        .model-id {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.8rem; cursor: pointer; user-select: all; transition: background 0.2s; }}
        .model-id:hover {{ background: #e5e7eb; }}
        .model-id.copied {{ background: #bbf7d0; color: #166534; }}
        
        a {{ color: #f38020; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }}
        .badge.deprecated {{ background: #fef3c7; color: #92400e; }}
        
        .copy-hint {{ font-weight: 400; color: #9ca3af; font-size: 0.75rem; }}
        
        .legend {{ display: flex; gap: 16px; margin-top: 24px; padding-top: 16px; border-top: 1px solid #e5e7eb; flex-wrap: wrap; font-size: 0.85rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 4px; }}
        
        @media (max-width: 900px) {{
            table {{ display: block; overflow-x: auto; }}
            .container {{ padding: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>☁️ Cloudflare Workers AI 模型比較表</h1>
        <p class="meta">資料來源: <a href="https://developers.cloudflare.com/workers-ai/models/" target="_blank">Cloudflare Workers AI Models</a> · <a href="https://developers.cloudflare.com/workers-ai/platform/pricing/" target="_blank">Pricing</a> · <a href="https://developers.cloudflare.com/workers-ai/platform/limits/" target="_blank">Limits</a><br>
        更新時間: {now} · 每日 05:00 UTC 自動更新 (GitHub Actions)</p>
        
        <div class="stats">
            <div class="stat-card free">
                <div class="stat-value">{free_count}</div>
                <div class="stat-label">免費方案可用模型</div>
            </div>
            <div class="stat-card paid">
                <div class="stat-value">{paid_count}</div>
                <div class="stat-label">需付費方案模型</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{free_count + paid_count}</div>
                <div class="stat-label">總模型數</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">10,000</div>
                <div class="stat-label">免費 Neurons/天</div>
            </div>
        </div>
        
        {render_table(free_models, "✅ 免費方案可用模型 (含免費 10,000 Neurons/天額度)")}
        
        {render_table(paid_models, "🔒 需付費方案模型 (需 Workers Paid 或 AI Gateway Credits)")}
        
        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background:#f0fdf4;border:1px solid #bbf7d0"></span> 免費模型 (綠色底)</div>
            <div class="legend-item"><span class="legend-color" style="background:#fef2f2;border:1px solid #fecaca"></span> 付費模型 (紅色底)</div>
            <div class="legend-item"><span class="badge deprecated">⚠ 已棄用/預覽</span> 模型狀態標示</div>
            <div class="legend-item">💡 <strong>免費額度</strong>: 每天 10,000 Neurons，約可換算為表中「免費額度估算」欄的請求次數 (假設單次 1K in + 1K out tokens)</div>
            <div class="legend-item">📋 <strong>Model ID</strong>: 點擊即可複製，直接貼上到您的 AI 提供商設定中</div>
        </div>
    </div>
    
    <script>
        // Copy Model ID to clipboard
        document.querySelectorAll('.model-id').forEach(el => {{
            el.addEventListener('click', async () => {{
                const text = el.textContent.trim();
                try {{
                    await navigator.clipboard.writeText(text);
                    el.classList.add('copied');
                    el.title = '已複製!';
                    setTimeout(() => {{ el.classList.remove('copied'); el.title = '點擊複製'; }}, 2000);
                }} catch (e) {{
                    // Fallback
                    const ta = document.createElement('textarea');
                    ta.value = text;
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                    el.classList.add('copied');
                    setTimeout(() => el.classList.remove('copied'), 2000);
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
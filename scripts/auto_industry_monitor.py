from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "monitor_config.json"
OUTPUT_DIR = ROOT / "outputs" / "auto-monitor"
WORK_DIR = ROOT / "work"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


@dataclass
class RawItem:
    source_group: str
    source_name: str
    source_type: str
    title: str
    url: str
    source: str
    published: str
    summary: str
    matched_query: str
    brand: str | None


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def format_struct_time(struct_time: object) -> str:
    if not struct_time:
        return ""
    try:
        return datetime(*struct_time[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""


def detect_brand(text: str, brands: list[str]) -> str | None:
    lowered = text.lower()
    for brand in brands:
        if brand.lower() in lowered:
            return brand
    return None


def normalize_url(url: str) -> str:
    return url.split("&")[0].strip()


def build_google_news_rss_url(query: str, language: str, region: str) -> str:
    if language == "zh-CN":
        ceid = "CN:zh-Hans"
    else:
        ceid = f"{region}:{language}"
    return (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl={language}&gl={region}&ceid={quote_plus(ceid)}"
    )


def fetch_feed(url: str, session: requests.Session) -> feedparser.FeedParserDict:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return feedparser.parse(response.content)


def expand_query_templates(group: dict[str, Any]) -> list[str]:
    keywords = group.get("keywords", [])
    domains = group.get("domains", [])
    queries: list[str] = []

    for keyword in keywords:
        queries.append(keyword)
        for domain in domains:
            queries.append(f"{keyword} site:{domain}")
    return queries


def collect_raw_items(config: dict[str, Any]) -> list[RawItem]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen_urls: set[str] = set()
    raw_items: list[RawItem] = []
    brands = config.get("brands", [])
    max_items = int(config.get("max_items_per_query", 8))

    for source_group in config.get("source_groups", []):
        group_name = source_group["name"]
        source_type = source_group.get("type", "search")
        queries = expand_query_templates(source_group)

        for language_profile in source_group.get("language_profiles", config.get("language_profiles", [])):
            language = language_profile["language"]
            region = language_profile["region"]

            for query in queries:
                rss_url = build_google_news_rss_url(query, language=language, region=region)
                feed = fetch_feed(rss_url, session)

                for entry in feed.entries[:max_items]:
                    url = normalize_url(entry.get("link", ""))
                    if not url or url in seen_urls:
                        continue

                    title = clean_text(entry.get("title", ""))
                    summary = clean_text(entry.get("summary", ""))[:400]
                    published = format_struct_time(entry.get("published_parsed"))
                    source = clean_text(entry.get("source", {}).get("title", "")) or "Unknown"
                    brand = detect_brand(f"{title} {summary}", brands)

                    raw_items.append(
                        RawItem(
                            source_group=group_name,
                            source_name=group_name,
                            source_type=source_type,
                            title=title,
                            url=url,
                            source=source,
                            published=published,
                            summary=summary,
                            matched_query=query,
                            brand=brand,
                        )
                    )
                    seen_urls.add(url)

                time.sleep(1)

    return raw_items


def fallback_category(item: RawItem) -> str:
    text = f"{item.title} {item.summary}".lower()

    if any(word in text for word in ["launch", "debut", "上市", "发布", "首发", "new model", "facelift"]):
        return "新车上市"
    if any(
        word in text
        for word in ["sales", "deliveries", "销量", "交付", "monthly sales", "annual sales", "half-year"]
    ):
        return "市场销量"
    if any(
        word in text
        for word in ["campaign", "marketing", "partnership", "联名", "营销", "传播", "赞助", "activation"]
    ):
        return "重要营销动态"
    return "舆情监控"


def fallback_analysis(config: dict[str, Any], raw_items: list[RawItem]) -> dict[str, Any]:
    categories = config.get("analysis_categories", [])
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in categories}

    for item in raw_items:
        category = fallback_category(item)
        grouped.setdefault(category, []).append(
            {
                "title": item.title,
                "url": item.url,
                "source": item.source,
                "published": item.published,
                "brand": item.brand,
                "source_group": item.source_group,
                "reason": f"Matched by fallback rule from query: {item.matched_query}",
                "summary": item.summary,
            }
        )

    return {
        "generated_by": "fallback",
        "executive_summary": [
            f"Collected {len(raw_items)} items from {len(config.get('source_groups', []))} source groups.",
            "DeepSeek API key was not configured, so this report uses local heuristic classification.",
        ],
        "categories": [
            {
                "name": category_name,
                "takeaways": [f"{len(grouped.get(category_name, []))} items found."],
                "items": grouped.get(category_name, []),
            }
            for category_name in categories
        ],
    }


def build_deepseek_prompt(config: dict[str, Any], raw_items: list[RawItem]) -> str:
    categories = ", ".join(config.get("analysis_categories", []))
    source_groups = ", ".join(group["name"] for group in config.get("source_groups", []))
    payload = [
        {
            "title": item.title,
            "summary": item.summary,
            "url": item.url,
            "source": item.source,
            "published": item.published,
            "brand": item.brand,
            "source_group": item.source_group,
        }
        for item in raw_items[: config.get("llm_item_limit", 80)]
    ]
    return (
        "You are an automotive industry analyst. "
        "Analyze the collected items and return valid JSON only. "
        "The source groups are: "
        f"{source_groups}. "
        "Classify every relevant item into one of these categories: "
        f"{categories}. "
        "Return this schema exactly: "
        '{"executive_summary":["..."],'
        '"categories":[{"name":"...", "takeaways":["..."], '
        '"items":[{"title":"...", "url":"...", "source":"...", "published":"...", '
        '"brand":"...", "source_group":"...", "reason":"...", "summary":"..."}]}]}. '
        "Each item summary should be around 80 to 120 Chinese characters and explain the key fact clearly. "
        "For market sales, explicitly call out star models and monthly, annual, or half-year sales mentions when present. "
        "For public opinion monitoring, prioritize controversy, complaints, safety, pricing, and regulatory attention. "
        "Keep the original source URL for each item. "
        "Input items JSON: "
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def call_deepseek(config: dict[str, Any], raw_items: list[RawItem]) -> dict[str, Any] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL") or config.get("deepseek_model", "deepseek-chat")
    prompt = build_deepseek_prompt(config, raw_items)
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You return strict JSON and do not add markdown fences.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    data = parse_llm_json(content)
    data["generated_by"] = model
    return data


def parse_llm_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = content[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    error_dump_path = WORK_DIR / "deepseek-invalid-response.txt"
    error_dump_path.write_text(content, encoding="utf-8")
    raise json.JSONDecodeError("Unable to parse LLM response as JSON", content, 0)


def build_markdown_report(config: dict[str, Any], analysis: dict[str, Any], generated_at: str) -> str:
    report_date = generated_at.split("T", 1)[0]
    lines = [
        f"# Automotive Industry Daily Monitor - {report_date}",
        "",
        f"- Generated at: {generated_at} UTC",
        f"- Analysis engine: {analysis.get('generated_by', 'unknown')}",
        f"- Category set: {', '.join(config.get('analysis_categories', []))}",
        "",
        "## Executive Summary",
        "",
    ]

    for point in analysis.get("executive_summary", []):
        lines.append(f"- {point}")

    for category in analysis.get("categories", []):
        lines.extend(["", f"## {category['name']}", ""])
        for takeaway in category.get("takeaways", []):
            lines.append(f"- {takeaway}")

        items = category.get("items", [])
        if not items:
            lines.append("- No items found.")
            continue

        for item in items:
            detail = f"Source: {item.get('source', 'Unknown')}"
            if item.get("brand"):
                detail += f" | Brand: {item['brand']}"
            if item.get("published"):
                detail += f" | Published: {item['published']}"
            if item.get("source_group"):
                detail += f" | Group: {item['source_group']}"

            lines.append(f"- [{item.get('title', 'Untitled')}]({item.get('url', '#')})")
            lines.append(f"  {detail}")
            if item.get("reason"):
                lines.append(f"  Why it matters: {item['reason']}")
            if item.get("summary"):
                lines.append(f"  Summary: {item['summary']}")

    return "\n".join(lines) + "\n"


def write_outputs(config: dict[str, Any], raw_items: list[RawItem], analysis: dict[str, Any]) -> tuple[Path, Path, Path]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report_date = generated_at.split("T", 1)[0]
    output_folder = OUTPUT_DIR / report_date
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_path = output_folder / "raw_items.json"
    analysis_path = output_folder / "analysis.json"
    report_path = output_folder / "report.md"

    raw_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "item_count": len(raw_items),
                "items": [asdict(item) for item in raw_items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown_report(config, analysis, generated_at), encoding="utf-8")
    return raw_path, analysis_path, report_path


def shorten_text(text: str, target: int = 110) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= target:
        return cleaned
    return cleaned[: target - 1].rstrip() + "…"


def build_feishu_text(analysis: dict[str, Any]) -> str:
    lines = ["汽车行业监控日报", ""]

    for point in analysis.get("executive_summary", [])[:3]:
        lines.append(f"摘要：{point}")

    for category in analysis.get("categories", [])[:4]:
        category_name = category.get("name", "未命名分类")
        items = category.get("items", [])
        lines.extend(["", f"【{category_name}】共 {len(items)} 条"])

        if not items:
            lines.append("暂无相关内容")
            continue

        for index, item in enumerate(items, start=1):
            title = item.get("title", "未命名资讯")
            summary = shorten_text(item.get("summary") or item.get("reason") or "暂无概述")
            url = item.get("url", "")
            lines.append(f"{index}. {title}")
            lines.append(f"概述：{summary}")
            if url:
                lines.append(f"链接：{url}")
            if item.get("source"):
                lines.append(f"来源：{item['source']}")
            lines.append("")

    return "\n".join(lines).strip()


def build_feishu_sign(secret: str, timestamp: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def send_feishu_message(analysis: dict[str, Any]) -> None:
    webhook = os.getenv("FEISHU_BOT_WEBHOOK", "").strip()
    if not webhook:
        return

    payload: dict[str, Any] = {
        "msg_type": "text",
        "content": {
            "text": build_feishu_text(analysis),
        },
    }

    secret = os.getenv("FEISHU_BOT_SECRET", "").strip()
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = build_feishu_sign(secret, timestamp)

    response = requests.post(webhook, json=payload, timeout=30)
    response.raise_for_status()


def main() -> int:
    config = load_config()
    try:
        raw_items = collect_raw_items(config)
        analysis = call_deepseek(config, raw_items) or fallback_analysis(config, raw_items)
        raw_path, analysis_path, report_path = write_outputs(config, raw_items, analysis)
        send_feishu_message(analysis)
    except requests.RequestException as exc:
        print(f"Network request failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"DeepSeek response was not valid JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Raw data written to: {raw_path}")
    print(f"Analysis written to: {analysis_path}")
    print(f"Report written to: {report_path}")
    print(f"Collected {len(raw_items)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

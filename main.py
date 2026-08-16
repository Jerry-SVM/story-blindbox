"""
故事盲盒 后端服务
对接 OpenAI 兼容 API（DeepSeek 等），提供大纲生成 + SSE 流式故事生成两个接口。
"""
import os
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ============ 初始化 OpenAI 兼容客户端（懒加载） ============
# 懒加载：避免 API_KEY 未配置时 import 阶段直接崩溃（Render 上无 .env）
_client: AsyncOpenAI | None = None
MODEL = os.getenv("MODEL", "deepseek-chat")


def _get_client() -> AsyncOpenAI:
    """首次调用接口时才创建 client，Render 环境变量配置完后不会有问题。"""
    global _client
    if _client is None:
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="API_KEY 未配置，请在 Render 后台 Environment 中填写。")
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        )
    return _client

# 项目根目录（backend/ 的上一级，放 index.html 的地方）
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR

app = FastAPI(title="故事盲盒")

# ============ CORS 配置 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 挂载前端静态文件（index.html） ============
# 部署到 Render 时，前后端共用一个域名，避免跨域与 API_BASE 配置问题
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    """根路径直接返回 index.html，用户访问 https://xxx.onrender.com 就能看到页面。"""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/index.html")
async def index_html():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ============ 请求体模型 ============
class OutlineRequest(BaseModel):
    keywords: list[str]  # 三个关键词
    style: str           # 文风
    wordCount: int       # 目标字数


class StoryRequest(BaseModel):
    outline: dict            # 接口1返回的大纲 JSON
    only_twist: bool = False # 仅重新生成反转结局


# ============ 接口1：生成大纲 POST /api/outline ============
@app.post("/api/outline")
async def generate_outline(req: OutlineRequest):
    # 大纲 JSON 模板（单独定义避免与 f-string 引号冲突）
    json_format = '{"title": "标题", "chapters": [{"title": "章节名", "summary": "20字梗概"}], "twist": "反转结局的设定"}'
    system_prompt = (
        # —— 纠偏指令：自定义文风可能含无关内容，模型自行判断 ——
        '注意：用户输入的"自定义文风"字段（custom_style）可能包含与写作风格完全无关的内容，'
        '如日常闲聊（"今天天气真好"）、无意义词汇等。如果该内容与小说风格、氛围、情绪、文笔无关，'
        '请自动忽略该字段，默认采用"现代文艺治愈风"进行创作。不要向用户提问或指出问题，直接按默认风格生成大纲。'
        # —— 原有指令（保持不变） ——
        "你是一位擅长埋伏笔的短篇小说大师。"
        f"请根据用户提供的三个关键词：{req.keywords}，"
        f"以{req.style}的文风，生成一个800-2000字微小说的详细大纲。"
        f"必须输出JSON格式：{json_format}。"
        "严禁生成色情、暴力及违反中国法律法规的内容。"
    )
    try:
        resp = await _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"},  # 强制 JSON 输出
            temperature=0.8,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        # 透传 wordCount，供 /api/story 分配各章字数
        data["wordCount"] = req.wordCount
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="模型返回的不是合法 JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用模型失败：{e}")


# ============ 接口2：生成故事 POST /api/story（SSE 流式） ============
@app.post("/api/story")
async def generate_story(req: StoryRequest):
    outline = req.outline
    only_twist = req.only_twist

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            # 人物一致性约束（每章扩写提示词必含）
            consistency = '请确保人物姓名、性格前后一致，禁止出现"他/她"指代不明的情况。'

            async def stream_one(prompt: str, chapter: int):
                """单次流式生成并推送。chapter>0 表示正文章节，-1 表示重写的结局。"""
                stream = await _get_client().chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.9 if chapter == -1 else 0.85,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        # SSE 格式：data: <json>\n\n
                        payload = {"text": delta, "chapter": chapter}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

            if only_twist:
                # 仅重新生成反转结局
                prompt = (
                    f"小说标题：{outline.get('title', '')}。"
                    f"这是该小说的反转结局设定：{outline.get('twist', '')}。"
                    "请基于此写一段全新的结局正文（约200-300字），承接前文情绪。"
                    + consistency
                    + "直接输出正文，不要加标题，不要任何解释。"
                )
                async for piece in stream_one(prompt, -1):
                    yield piece
            else:
                # 完整生成：逐章扩写
                chapters = outline.get("chapters", [])
                for idx, ch in enumerate(chapters):
                    prompt = (
                        f"小说标题：{outline.get('title', '')}。"
                        f"当前正在写第{idx+1}章「{ch.get('title', '')}」，"
                        f"本章梗概：{ch.get('summary', '')}。"
                        f"全文目标约{outline.get('wordCount', '未知')}字，请合理分配本章字数。"
                        + consistency
                        + "直接输出本章正文，不要加章节标题，不要任何解释。"
                    )
                    async for piece in stream_one(prompt, idx + 1):
                        yield piece
            # 结束标记
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n".encode("utf-8")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，保证实时推送
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

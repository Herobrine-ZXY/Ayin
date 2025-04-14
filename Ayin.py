#本程序（即此程序与同目录的的所有文件）均暂不开源（其实是我程序太史山了），如果你未经作者授权看到此条消息请立即退出。                 作者：PAZJ
import threading
import hmac
import base64
import time
import uuid
import requests
import json
import io
import logging
import os
import glob
import tkinter as tk
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
from openai import OpenAI
from hashlib import sha1

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ai_assistant.log'
)
logger = logging.getLogger(__name__)


def read_api_keys():
    """从API.key文件中读取API密钥"""
    try:
        with open("API.key", "r") as f:
            lines = [line.strip() for line in f.readlines()]

        if len(lines) < 3:
            raise ValueError("API.key文件至少会包含3行内容，排列顺序为：1.deep seek的API密钥   2.星火的API签名密钥   3.星火的API访问密钥")

        return {
            "deepseek_key": lines[0],  # 第1行: DeepSeek API密钥
            "image_api_key": lines[1],  # 第2行: 图像API密钥
            "signature_secret": lines[2],  # 第3行: 签名密钥
            "base_url": "https://api.deepseek.com"  # 固定URL
        }
    except Exception as e:
        logger.error(f"读取API密钥失败: {str(e)}")
        raise


# 清理20天前的日志文件
def clean_old_logs(days_to_keep=20):
    """清理指定天数前的日志文件"""
    try:
        log_files = glob.glob('ai_assistant.log*')  # 获取所有日志文件

        for log_file in log_files:
            # 获取文件修改时间
            file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            # 计算文件存在天数
            file_age = datetime.now() - file_mtime

            if file_age > timedelta(days=days_to_keep):
                try:
                    os.remove(log_file)
                    logger.info(f"已删除旧日志文件: {log_file} (创建于{file_age.days}天前)")
                except Exception as e:
                    logger.error(f"删除旧日志文件失败: {log_file}, 错误: {str(e)}")
    except Exception as e:
        logger.error(f"清理旧日志时发生错误: {str(e)}")


# 程序启动时自动清理旧日志
clean_old_logs()

# 读取API密钥
try:
    api_keys = read_api_keys()
except Exception as e:
    logger.error(f"无法读取API密钥: {str(e)}")
    messagebox.showerror("错误", f"无法读取API密钥: {str(e)}")
    exit(1)

# 初始化 OpenAI 客户端
try:
    client = OpenAI(
        api_key=api_keys["deepseek_key"],
        base_url=api_keys["base_url"],
        timeout=30  # 设置超时时间
    )
except Exception as e:
    logger.error(f"初始化OpenAI客户端失败: {str(e)}")
    raise

# 初始化对话历史
messages: List[Dict[str, str]] = [
    {"role": "system", "content": "你是一个助手，请用中文回答用户的问题。"}
]

# 错误码与解释的映射
error_explanations: Dict[int, str] = {
    400: "格式错误",
    401: "认证失败",
    402: "余额不足",
    422: "参数错误",
    429: "请求速率达到上限",
    500: "服务器故障",
    503: "服务器繁忙"
}

# 错误码与详细解决方案的映射
error_solutions: Dict[int, str] = {
    400: "原因：请求体格式错误\n解决方法：请根据错误信息提示修改请求体",
    401: "原因：API key 错误，认证失败\n解决方法：请检查您的 API key 是否正确，如没有 API key，请先创建 API key",
    402: "原因：账号余额不足\n解决方法：请确认账户余额，并前往 https://platform.deepseek.com/sign_in 的充值页面进行充值",
    422: "原因：请求体参数错误\n解决方法：请根据错误信息提示修改相关参数",
    429: "原因：请求速率（TPM 或 RPM）达到上限\n解决方法：请合理规划您的请求速率",
    500: "原因：服务器内部故障\n解决方法：请等待后重试。若问题一直存在，请联系我们解决",
    503: "原因：服务器负载过高\n解决方法：请稍后重试您的请求"
}

# 文生图API配置
API_KEY = api_keys["image_api_key"]
API_SECRET = api_keys["signature_secret"]  # 签名密钥
API_URL = "https://www.liblib.art/modelinfo/f8b990b20cb943e3aa0e96f34099d794?versionUuid=21df5d84cca74f7a885ba672b5a80d19"
REQUEST_TIMEOUT = 30  # 请求超时时间(秒)
IMAGE_SAVE_DIR = "generated_images"  # 图片保存目录


class ImageGenerationError(Exception):
    """自定义图像生成异常"""
    pass


def make_sign(uri: str = "/api/genImg") -> Tuple[str, str, str]:
    """生成签名"""
    try:
        timestamp = str(int(time.time() * 1000))
        signature_nonce = str(uuid.uuid4())
        content = '&'.join((uri, timestamp, signature_nonce))
        digest = hmac.new(API_SECRET.encode(), content.encode(), sha1).digest()
        sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
        return sign, timestamp, signature_nonce
    except Exception as e:
        logger.error(f"生成签名失败: {str(e)}")
        raise


def generate_image(prompt: str, model: str = "F.1") -> str:
    """调用文生图API生成图像"""
    try:
        # 生成签名
        sign, timestamp, nonce = make_sign()

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Signature": sign,
            "Timestamp": timestamp,
            "SignatureNonce": nonce
        }

        data = {
            "prompt": prompt,
            "model": model
        }

        logger.info(f"发送图像生成请求，提示词: {prompt}")
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        response_data = response.json()
        logger.debug(f"API响应: {response_data}")

        if not isinstance(response_data, dict):
            raise ImageGenerationError("API返回了无效的响应格式")

        image_url = response_data.get("image_url")
        if not image_url or not isinstance(image_url, str):
            raise ImageGenerationError("API响应中未包含有效的图像URL")

        return image_url

    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', 'N/A')
        error_msg = f"API请求失败: {str(e)} (状态码: {status_code})"
        logger.error(f"{error_msg}\n响应内容: {e.response.text if hasattr(e, 'response') else '无响应'}")
        return error_msg
    except json.JSONDecodeError:
        error_msg = "API返回了无效的JSON响应"
        logger.error(error_msg)
        return error_msg
    except ImageGenerationError as e:
        error_msg = f"图像生成错误: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


def show_image_in_ui(img_data: bytes) -> str:
    """在UI中显示图像"""
    try:
        # 清除之前的图像
        for item in canvas.find_withtag("image_tag"):
            canvas.delete(item)

        # 加载图像
        img = Image.open(io.BytesIO(img_data))
        # 调整图像大小以适应界面
        img.thumbnail((600, 600))

        # 转换为Tkinter可用的格式
        photo = ImageTk.PhotoImage(img)

        # 在Canvas上显示图像
        canvas.create_image(1200, 200, anchor="nw", image=photo, tags="image_tag")
        canvas.image = photo  # 保持引用

        return "图像已显示在界面中"
    except Exception as e:
        error_msg = f"图像显示失败: {str(e)}"
        logger.error(error_msg)
        return error_msg


def save_image_to_file(img_data: bytes, prompt: str) -> str:
    """保存图片到文件"""
    try:
        # 创建保存目录（如果不存在）
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

        # 生成文件名（使用时间戳和提示词前10个字符）
        timestamp = int(time.time())
        safe_prompt = "".join(c for c in prompt[:10] if c.isalnum())
        filename = f"{IMAGE_SAVE_DIR}/image_{timestamp}_{safe_prompt}.jpg"

        # 保存图片
        with open(filename, "wb") as f:
            f.write(img_data)

        return filename
    except Exception as e:
        error_msg = f"图片保存失败: {str(e)}"
        logger.error(error_msg)
        return error_msg


def show_image_help():
    """显示生成图片的帮助信息"""
    help_text = """生成图片帮助:
- 输入普通文本进行对话
- 输入'生成签名'获取API签名
- 输入'生成图[描述]'或'生成图像 描述'生成图片
  例如: 生成图[一只可爱的猫] 或 生成图像 一只可爱的猫
生成的图片会自动保存到程序目录下的 generated_images 文件夹中"""

    output_box.insert(tk.END, help_text + "\n")
    output_box.see(tk.END)


def safe_update_output(text: str):
    """安全更新输出框"""
    try:
        output_box.insert(tk.END, f"助手：{text}\n")
        output_box.see(tk.END)
    except Exception as e:
        logger.error(f"更新输出框失败: {str(e)}")


def process_image_generation(prompt: str, retries: int = 2):
    """处理图像生成流程"""
    try:
        for attempt in range(retries + 1):
            image_url = generate_image(prompt)
            logger.info(f"第{attempt + 1}次尝试，图像生成结果: {image_url}")

            if not isinstance(image_url, str):
                if attempt == retries:
                    safe_update_output("图像生成失败: 无效的响应")
                    return
                continue

            if image_url.startswith("http"):
                try:
                    # 下载图片
                    response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    img_data = response.content

                    # 保存图片到本地
                    saved_path = save_image_to_file(img_data, prompt)
                    if saved_path.startswith(IMAGE_SAVE_DIR):
                        save_msg = f"图片已保存到: {saved_path}"
                    else:
                        save_msg = saved_path  # 错误信息

                    # 在UI中显示
                    display_result = show_image_in_ui(img_data)

                    safe_update_output(
                        f"图像生成成功！\n"
                        f"{display_result}\n"
                        f"{save_msg}\n"
                        f"提示词: {prompt}"
                    )
                    return
                except Exception as e:
                    if attempt == retries:
                        error_msg = f"图像下载失败: {str(e)}"
                        safe_update_output(error_msg)
                        logger.error(error_msg)
                    continue
            else:
                if attempt == retries:
                    safe_update_output(f"图像生成失败: {image_url}")
                continue

    except Exception as e:
        safe_update_output(f"处理图像生成时发生错误: {str(e)}")
        logger.error(f"处理图像生成时发生错误: {str(e)}")


def fetch_assistant_reply(user_input: str):
    """获取助手回复"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            timeout=30
        )

        if response.choices:
            assistant_reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_reply})
            safe_update_output(assistant_reply)
    except Exception as e:
        error_code = getattr(e, "status_code", None)
        if error_code in error_explanations:
            explanation = f"错误码 {error_code} - {error_explanations[error_code]}"
            safe_update_output(explanation)
            if "详细信息" in user_input or "解决方法" in user_input:
                solution = f"错误码 {error_code} - {error_solutions[error_code]}"
                safe_update_output(solution)
        else:
            safe_update_output(f"错误：{e}")
        logger.error(f"获取助手回复失败: {str(e)}")


def send_message(event=None):
    """发送消息处理函数"""
    user_input = input_box.get("1.0", tk.END).strip()
    input_box.delete("1.0", tk.END)

    if not user_input:
        return

    messages.append({"role": "user", "content": user_input})
    output_box.insert(tk.END, f"你：{user_input}\n")
    output_box.see(tk.END)

    # 检查特殊命令
    if any(cmd in user_input.lower() for cmd in ["生成图", "生成图像", "生成图片"]):
        prompt = user_input.split("]", 1)[-1].strip() if "]" in user_input else user_input.split(" ", 1)[-1].strip()
        if not prompt:
            safe_update_output("请提供图像描述")
            return

        safe_update_output("正在生成图像...")
        threading.Thread(
            target=process_image_generation,
            args=(prompt,),
            daemon=True
        ).start()
    elif user_input.lower() == "生成签名":
        try:
            sign, timestamp, nonce = make_sign()
            output_box.insert(tk.END, f"签名信息：\n签名: {sign}\n时间戳: {timestamp}\n随机数: {nonce}\n")
            output_box.see(tk.END)
        except Exception as e:
            safe_update_output(f"生成签名失败: {str(e)}")
    elif user_input.lower() == "生成图片帮助":
        show_image_help()
    elif user_input.lower() == "测试图片生成":
        test_prompt = "a cute cat"  # 简单英文提示
        safe_update_output(f"测试图片生成，使用提示: {test_prompt}")
        threading.Thread(
            target=process_image_generation,
            args=(test_prompt,),
            daemon=True
        ).start()
    else:
        safe_update_output("加载中...")
        threading.Thread(
            target=fetch_assistant_reply,
            args=(user_input,),
            daemon=True
        ).start()


def insert_newline(event=None):
    """插入换行符"""
    input_box.insert(tk.END, "\n")
    return "break"


def on_closing():
    """窗口关闭事件处理"""
    if messagebox.askokcancel("退出", "确定要退出程序吗？"):
        root.destroy()


# 创建主窗口
root = tk.Tk()
root.title("AI 对话助手")
root.geometry("1920x1080")
root.protocol("WM_DELETE_WINDOW", on_closing)

# 加载背景图片
try:
    bg_image = Image.open("UI.png")
    bg_photo = ImageTk.PhotoImage(bg_image)
    canvas = tk.Canvas(root, width=1920, height=1080)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")
except Exception as e:
    logger.error(f"无法加载背景图片: {str(e)}")
    canvas = tk.Canvas(root, width=1920, height=1080, bg="#1e1e1e")
    canvas.pack(fill="both", expand=True)

# 创建输出框和输入框
output_box = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=100,
    height=30,
    bd=0,
    highlightthickness=0,
    bg="#2d2d2d",
    fg="white",
    insertbackground="white"
)
canvas.create_window(100, 100, anchor="nw", window=output_box, width=800, height=600)

input_box = tk.Text(
    root,
    height=5,
    width=100,
    bd=0,
    highlightthickness=0,
    bg="#2d2d2d",
    fg="white",
    insertbackground="white"
)
canvas.create_window(100, 720, anchor="nw", window=input_box, width=800, height=150)

# 绑定按键事件
input_box.bind("<Return>", send_message)
input_box.bind("<Shift-Return>", insert_newline)

# 运行主循环
root.mainloop()

# coding=utf-8
import threading
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import scrolledtext, messagebox, filedialog, simpledialog, Frame, Label, Button, OptionMenu, StringVar, \
    Entry
from typing import Dict, List, Optional
import logging
import os
import glob
import time
import json
import requests
from openai import OpenAI
from PIL import Image, ImageTk
import speech_recognition as sr
import io
import vt
import re
import socket
import traceback

# ====================== 配置部分 ======================
CEPHALON_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJJbmZvIjoiYXBpOjE5MzM3MDk3Nzk4NzI4NjI3ODQiLCJleHAiOjE3ODE0MDM3MDV9.kQVb_LvvNDcNQ5ddpgKmZXgu5wmvFuiBz455w3o4ZKI"
VIRUSTOTAL_API_KEY = "ad89fe2131ab47c3c3188056b871814f5b3f8b81425902d09508f59493467f5e"
COMFYUI_API_URL = "https://cephalon.cloud/user-center/v1/model/comfyui"
MODEL_ID = "1854732937730371541"  # FLUX.1模型ID

# 创建OpenAI客户端
openai_client = OpenAI(
    api_key=CEPHALON_API_KEY,
    base_url="https://cephalon.cloud/user-center/v1/model"
)

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model.log'
)
logger = logging.getLogger(__name__)

# ====================== 全局状态 ======================
messages = []  # 对话历史
HISTORY_FILE = "conversation_history.json"


# ====================== 工具函数 ======================
def is_internet_available():
    """检查网络连接是否可用"""
    try:
        # 使用更可靠的检查方法
        socket.create_connection(("1.1.1.1", 53), timeout=3)
        return True
    except OSError:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False


def load_conversation_history():
    """加载对话历史"""
    global messages
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                logger.info(f"已加载历史对话，共{len(messages)}条记录")
                # 确保至少有系统消息
                if len(messages) == 0:
                    messages = [{"role": "system", "content": "你是一个助手，请用中文回答用户的问题。"}]
                return True
    except Exception as e:
        logger.error(f"加载对话历史失败: {str(e)}")
        # 输出完整堆栈信息
        logger.error(traceback.format_exc())

    # 初始化默认对话
    messages = [{"role": "system", "content": "你是一个助手，请用中文回答用户的问题。"}]
    return False


def save_conversation_history():
    """保存对话历史"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            return True
    except Exception as e:
        logger.error(f"保存对话历史失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def clean_old_files(days_to_keep=20):
    """清理旧文件"""
    try:
        files_to_clean = glob.glob('*.log') + glob.glob('*.json') + glob.glob('generated_*.png')

        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_age = datetime.now() - file_mtime

                    if file_age > timedelta(days=days_to_keep):
                        os.remove(file_path)
                        logger.info(f"已删除旧文件: {file_path} (创建于{file_age.days}天前)")
            except Exception as e:
                logger.warning(f"无法删除文件 {file_path}: {str(e)}")
        return True
    except Exception as e:
        logger.error(f"清理旧文件时发生错误: {str(e)}")
        return False


def log_error(msg):
    """记录并显示错误"""
    logger.error(msg)
    logger.error(traceback.format_exc())
    output_box.insert(tk.END, f"[错误] {msg}\n", "error")
    output_box.see(tk.END)


def log_info(msg):
    """记录并显示信息"""
    logger.info(msg)
    output_box.insert(tk.END, f"[信息] {msg}\n")
    output_box.see(tk.END)


def log_warning(msg):
    """记录并显示警告"""
    logger.warning(msg)
    output_box.insert(tk.END, f"[警告] {msg}\n", "warning")
    output_box.see(tk.END)


def is_chinese(text):
    """检查文本是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def translate_to_english(text):
    """使用OpenAI API将中文翻译为英文"""
    if not is_chinese(text):
        return text

    log_info(f"正在翻译提示词: {text}")

    try:
        # 使用新版OpenAI API进行翻译
        response = openai_client.chat.completions.create(
            model="DeepSeek-R1",
            messages=[
                {"role": "system", "content": "你是一个专业的翻译助手，将中文提示词准确翻译为英文。"},
                {"role": "user", "content": f"请将以下图像描述从中文翻译为英文，保持艺术表达的准确性和完整性：{text}"}
            ],
            temperature=0.1,
            max_tokens=100
        )

        # 提取翻译结果
        translation = response.choices[0].message.content.strip()

        # 移除翻译说明（如"翻译为英文："）
        if ":" in translation:
            translation = translation.split(":", 1)[1].strip()

        log_info(f"翻译结果: {translation}")
        return translation
    except Exception as e:
        log_error(f"翻译失败: {str(e)}")
        return text  # 失败时返回原文


# ====================== 病毒扫描功能 ======================
def scan_file_with_virustotal():
    """使用VirusTotal扫描文件"""
    file_path = filedialog.askopenfilename(
        title="选择要扫描的文件",
        filetypes=[("所有文件", "*.*")]
    )

    if not file_path:
        return

    log_info(f"开始扫描文件: {os.path.basename(file_path)}")

    threading.Thread(
        target=perform_file_scan,
        args=(file_path,),
        daemon=True
    ).start()


def scan_url_with_virustotal():
    """使用VirusTotal扫描URL"""
    url = simpledialog.askstring("扫描URL", "请输入要扫描的URL:")

    if not url:
        return

    # 验证URL格式
    if not re.match(r'^https?://', url):
        messagebox.showwarning("格式错误", "URL必须以http://或https://开头")
        return

    if not is_internet_available():
        log_error("无法连接到互联网，请检查网络连接")
        return

    log_info(f"开始扫描URL: {url}")

    threading.Thread(
        target=perform_url_scan,
        args=(url,),
        daemon=True
    ).start()


def perform_file_scan(file_path: str):
    """执行文件扫描任务"""
    try:
        # 文件大小检查
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        if file_size > 32:
            log_error("文件过大，最大支持32MB")
            return

        if not is_internet_available():
            log_error("无法连接到互联网，请检查网络连接")
            return

        # 使用较短的超时设置
        with vt.Client(VIRUSTOTAL_API_KEY, timeout=15) as client:
            # 打开文件并扫描
            with open(file_path, "rb") as f:
                # 使用wait_for_completion等待扫描完成
                try:
                    analysis = client.scan_file(f, wait_for_completion=True)
                except vt.APIError as e:
                    if e.code == 429:
                        log_error("API请求过于频繁，请稍后再试")
                    else:
                        log_error(f"扫描错误: {str(e)}")
                    return
                except Exception as e:
                    log_error(f"扫描过程出错: {极(e)}")
                    return

                # 获取文件报告
                try:
                    file_report = client.get_object(f"/files/{analysis.id}")
                except vt.APIError as e:
                    log_error(f"获取报告失败: {str(e)}")
                    return

                # 解析扫描结果
                stats = file_report.last_analysis_stats
                malicious = stats.get("malicious", 0)
                suspicious = stats.get("suspicious", 0)
                harmless = stats.get("harmless", 0)
                undetected = stats.get("undetected", 0)
                timeout = stats.get("timeout", 0)
                total = malicious + suspicious + harmless + undetected + timeout

                # 安全评估
                safety_status = "安全"
                status_color = "success"
                if malicious > 5:
                    safety_status = "高危！"
                    status_color = "error"
                elif malicious > 0:
                    safety_status = "警告"
                    status_color = "warning"
                elif suspicious > 3:
                    safety_status = "可疑"
                    status_color = "warning"

                # 显示结果
                result_text = (
                    f"文件扫描完成: {os.path.basename(file_path)}\n"
                    f"安全状态: {safety_status}\n"
                    f"恶意引擎数: {malicious}/{total}\n"
                    f"可疑引擎数: {suspicious}/{total}\n"
                    f"无害引擎数: {harmless}/{total}\n"
                    f"报告链接: https://www.virustotal.com/gui/file/{analysis.id}\n"
                )

                output_box.insert(tk.END, result_text, status_color)

                # 添加安全状态标签
                if malicious > 0 or suspicious > 0:
                    output_box.insert(tk.END, "警告: 文件可能包含恶意内容!\n", "warning")
                else:
                    output_box.insert(tk.END, "文件安全\n", "success")

                output_box.see(tk.END)

    except Exception as e:
        log_error(f"文件扫描失败: {str(e)}")
        logger.error(traceback.format_exc())


def perform_url_scan(url: str):
    """执行URL扫描任务"""
    try:
        if not is_internet_available():
            log_error("无法连接到互联网，请检查网络连接")
            return

        # 使用较短的超时设置
        with vt.Client(VIRUSTOTAL_API_KEY, timeout=15) as client:
            # 扫描URL
            try:
                analysis = client.scan_url(url, wait_for_completion=True)
            except vt.APIError as e:
                if e.code == 429:
                    log_error("API请求过于频繁，请稍后再试")
                else:
                    log_error(f"扫描错误: {str(e)}")
                return
            except Exception as e:
                log_error(f"扫描过程出错: {str(e)}")
                return

            # 获取URL报告
            url_id = vt.url_id(url)
            try:
                url_report = client.get_object(f"/urls/{url_id}")
            except vt.APIError as e:
                log_error(f"获取报告失败: {str(e)}")
                return

            # 解析扫描结果
            stats = url_report.last_analysis_stats
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            harmless = stats.get("harmless", 0)
            undetected = stats.get("undetected", 0)
            total = malicious + suspicious + harmless + undetected

            # 安全评估
            safety_status = "安全"
            status_color = "success"
            if malicious > 5:
                safety_status = "高危！"
                status_color = "error"
            elif malicious > 0:
                safety_status = "警告"
                status_color = "warning"
            elif suspicious > 3:
                safety_status = "可疑"
                status_color = "warning"

            # 显示结果
            result_text = (
                f"URL扫描完成: {url}\n"
                f"安全状态: {safety_status}\n"
                f"恶意引擎数: {malicious}/{total}\n"
                f"可疑引擎数: {suspicious}/{total}\n"
                f"无害引擎数: {harmless}/{total}\n"
                f"报告链接: https://www.virustotal.com/gui/url/{url_id}\n"
            )

            output_box.insert(tk.END, result_text, status_color)

            # 添加安全状态标签
            if malicious > 0 or suspicious > 0:
                output_box.insert(tk.END, "警告: URL可能包含恶意内容!\n", "warning")
            else:
                output_box.insert(tk.END, "URL安全\n", "success")

            output_box.see(tk.END)

    except Exception as e:
        log_error(f"URL扫描失败: {str(e)}")
        logger.error(traceback.format_exc())


def show_virus_scan_menu():
    """显示病毒扫描菜单"""
    if not hasattr(root, '_virus_menu') or not root._virus_menu.winfo_exists():
        root._virus_menu = tk.Menu(root, tearoff=0)
        root._virus_menu.add_command(label="扫描文件", command=scan_file_with_virustotal)
        root._virus_menu.add_command(label="扫描URL", command=scan_url_with_virustotal)

    root._virus_menu.post(virus_button.winfo_rootx(), virus_button.winfo_rooty() + virus_button.winfo_height())


# ====================== 文生图功能 ======================
def on_size_change(*args):
    """当尺寸选择变化时更新显示的尺寸"""
    size = size_var.get()
    if size == "自定义尺寸":
        try:
            width = custom_width.get()
            height = custom_height.get()
            if width and height:
                size_display.set(f"{width}x{height}")
            else:
                size_display.set("自定义尺寸")
        except:
            size_display.set("自定义尺寸")
    else:
        size_map = size_options[size]
        size_display.set(f"{size_map['width']}x{size_map['height']}")


def on_custom_size_change(*args):
    """当自定义尺寸改变时更新尺寸显示和选择框"""
    try:
        width = custom_width.get()
        height = custom_height.get()
        if width and height:
            # 检查是否是预设尺寸
            for preset_name, preset_size in size_options.items():
                if preset_name != "自定义尺寸" and preset_size['width'] == int(width) and preset_size['height'] == int(
                        height):
                    size_var.set(preset_name)
                    return

            # 不是预设尺寸，设为自定义
            size_var.set("自定义尺寸")
            size_display.set(f"{width}x{height}")
    except ValueError:
        # 忽略无效输入
        pass


def generate_image_from_text():
    """调用ComfyUI API生成图像"""
    prompt = image_prompt.get("1.0", tk.END).strip()
    if not prompt:
        messagebox.showwarning("提示", "请输入提示词")
        return

    try:
        # 获取尺寸选择
        size = size极.get()
        if size == "自定义尺寸":
            width = int(custom_width.get())
            height = int(custom_height.get())
        else:
            size_map = size_options[size]
            width, height = size_map["width"], size_map["height"]

        # 高级参数
        steps = max(1, min(50, int(steps_var.get())))
        guidance = max(1.0, min(20.0, float(guidance_var.get())))
        seed = seed_var.get().strip()
        seed_value = int(seed) if seed.isdigit() else -1

    except ValueError as e:
        messagebox.showerror("参数错误", f"无效的参数值: {str(e)}")
        return

    # 记录开始时间
    start_time = time.time()
    log_info(f"开始生成图像: {prompt[:50]}...")

    # 在后台线程中执行API请求
    threading.Thread(
        target=perform_image_generation,
        args=(prompt, width, height, guidance, steps, seed_value, start_time),  # 修复：使用start_time而不是start极
        daemon=True
    ).start()


def perform_image_generation(prompt, width, height, guidance, steps, seed, start_time):
    """执行图像生成任务"""
    try:
        if not is_internet_available():
            log_error("无法连接到互联网，请检查网络连接")
            return

        # 检查提示词是否需要翻译
        if is_chinese(prompt):
            log_info("检测到中文提示词，正在进行翻译...")
            prompt = translate_to_english(prompt)
        else:
            log_info("使用英文提示词生成图像")

        # 准备请求数据
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance,
            "steps": steps,
            "seed": seed if seed != -1 else int(time.time() * 1000) % 10 ** 9
        }

        headers = {
            "Authorization": f"Bearer {CEPHALON_API_KEY}",
            "Model-Id": MODEL_ID,
            "Content-Type": "application/json"
        }

        # 发送API请求（超时设置为60秒）
        response = requests.post(
            COMFYUI_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            log_error(f"API错误: {response.status_code} - {response.text[:200]}")
            return

        # 检查内容类型
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            log_error(f"API返回非图像内容: {content_type}")
            return

        # 转换响应为图像
        try:
            image = Image.open(io.BytesIO(response.content))
        except Exception as e:
            log_error(f"图像解码失败: {str(e)}")
            return

        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"generated_{timestamp}.png"
        image.save(img_path)

        # 显示预览
        max_preview_size = 400
        ratio = min(max_preview_size / image.width, max_preview_size / image.height)
        preview_size = (int(image.width * ratio), int(image.height * ratio))
        preview = image.copy()
        preview.thumbnail(preview_size)
        photo = ImageTk.PhotoImage(preview)

        # 显示在输出框中
        output_box.image_create(tk.END, image=photo)
        output_box.insert(tk.END, f"\n[成功] 生成图像 ({elapsed:.1f}秒)\n", "success")
        output_box.insert(tk.END, f"保存为: {img_path}\n")
        output_box.see(tk.END)

        # 保持图像引用
        if not hasattr(output_box, 'image_refs'):
            output_box.image_refs = []
        output_box.image_refs.append(photo)

    except requests.exceptions.Timeout:
        log_error("图像生成超时（60秒）")
    except requests.exceptions.RequestException as e:
        log_error(f"请求失败: {str(e)}")
    except Exception as e:
        log_error(f"图像生成失败: {str(e)}")
        logger.error(traceback.format_exc())


# ====================== 对话功能 ======================
def safe_update_output(text: str, tag: Optional[str] = None):
    """安全更新输出框"""
    try:
        output_box.insert(tk.END, text, tag)
        output_box.see(tk.END)
        return True
    except Exception as e:
        log_error(f"更新输出框失败: {str(e)}")
        return False


def fetch_assistant_reply_stream(user_input: str):
    """流式获取助手回复 - 主要修复点"""
    try:
        # 添加用户消息到历史
        messages.append({"role": "user", "content": user_input})

        # 创建流式请求
        stream = openai_client.chat.completions.create(
            model="DeepSeek-R1",
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )

        # 初始化响应收集
        full_response = ""
        response_id = f"response_{time.time()}"

        # 显示初始提示
        safe_update_output("助手：", "assistant")

        # 处理流式响应
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                safe_update_output(content, "assistant")
                root.update()  # 强制更新UI以实时显示

        # 添加到对话历史
        if full_response:
            messages.append({"role": "assistant", "content": full_response})
            save_conversation_history()
            safe_update_output("\n", "assistant")

    except Exception as e:
        log_error(f"获取助手回复失败: {str(e)}")
        logger.error(traceback.format_exc())
        handle_api_error(e, user_input)


def handle_api_error(e, user_input):
    """处理API错误"""
    error_code = getattr(e, 'status_code', 500)
    explanation = error_explanations.get(error_code, "未知错误")
    solution = error_solutions.get(error_code, "请检查网络连接或稍后重试")

    error_msg = f"API错误 {error_code}: {explanation}\n{solution}"
    safe_update_output(error_msg, "error")

    # 如果用户询问错误详情，提供更多信息
    if "错误" in user_input or "问题" in user_input:
        safe_update_output(f"详细解决方案: {solution}")


def insert_newline(event=None):
    """插入换行符"""
    input_box.insert(tk.END, "\n")
    return "break"


def send_message(event=None):
    """发送消息处理函数"""
    user_input = input_box.get("1.0", tk.END).strip()
    input_box.delete("1.0", tk.END)

    if not user_input:
        return

    # 显示用户输入
    safe_update_output(f"用户：{user_input}\n", "user")

    # 流式获取助手回复
    threading.Thread(
        target=fetch_assistant_reply_stream,
        args=(user_input,),
        daemon=True
    ).start()


def show_help():
    """显示帮助信息"""
    help_text = """DeepSeek-R1 智能助手使用说明:

1. 对话功能
- 输入问题后按Enter或点击发送按钮
- 使用Shift+Enter换行
- 支持对话上下文记忆

2. 语音输入
- 点击"语音输入"按钮进行语音识别
- 使用Ctrl+Shift+S切换语音识别状态
- 自动修正常见错别字

3. 图像生成
- 支持中文提示词（自动翻译为英文）
- 尺寸选择器实时显示具体尺寸
- 自定义尺寸自动识别
- 点击"生成图像"创建图片

4. 病毒扫描
- 点击"病毒扫描"按钮扫描文件或URL
- 支持本地文件扫描和在线URL检测
- 显示详细安全报告
- 网络超时重试机制

5. 其他功能
- 点击"帮助"查看使用说明
- 退出程序: 点击窗口关闭按钮或按Alt+F4

技术支持:
- 对话API: cephalon.cloud
- 图像API: FLUX.1
- 病毒扫描: VirusTotal API
"""
    output_box.insert(tk.END, help_text + "\n")
    output_box.see(tk.END)


# ====================== 语音识别 ======================
class VoiceRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.stop_listening = None

    def start_listening(self):
        """开始语音识别"""
        if self.is_listening:
            return

        self.is_listening = True
        threading.Thread(target=self._listen_thread).start()

    def stop_listening_now(self):
        """停止语音识别"""
        if self.is_listening and self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            self.is_listening = False

    def _listen_thread(self):
        """语音识别后台线程"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                safe_update_output("语音识别已启动，请开始说话...\n")

                def callback(recognizer, audio):
                    try:
                        text = recognizer.recognize_google(audio, language='zh-CN')
                        self._process_recognized_text(text)
                    except sr.UnknownValueError:
                        safe_update_output("语音识别失败：无法理解音频\n")
                    except sr.RequestError as e:
                        safe_update_output(f"语音识别错误: {str(e)}\n")

                self.stop_listening = self.recognizer.listen_in_background(
                    source, callback, phrase_time_limit=10)
        except Exception as e:
            log_error(f"语音识别初始化失败: {str(e)}")

    def _process_recognized_text(self, text):
        """处理识别到的文本"""
        # 自动修正常见错别字
        corrections = {"在吗": "在吗", "干什": "干什么", "为什": "为什么"}
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)

        # 插入输入框
        input_box.delete("1.0", tk.END)
        input_box.insert(tk.END, text)

        # 自动发送
        send_message()


# ====================== 主界面 ======================
def toggle_image_panel():
    """切换图像面板显示状态"""
    if image_panel.winfo_ismapped():
        image_panel.pack_forget()
        image_button.config(text="显示图像设置")
    else:
        image_panel.pack(fill="x", padx=10, pady=5, after=output_frame)
        image_button.config(text="隐藏图像设置")


def toggle_voice_recognition():
    """切换语音识别状态"""
    if voice_recognition.is_listening:
        voice_recognition.stop_listening_now()
        voice_button.config(text="开始语音输入")
        safe_update_output("语音识别已停止\n")
    else:
        voice_recognition.start_listening()
        voice_button.config(text="停止语音输入")


def on_closing():
    """窗口关闭事件处理"""
    if messagebox.askokcancel("退出", "确定要退出程序吗？"):
        save_conversation_history()
        root.destroy()


# ====================== 初始化应用 ======================
root = tk.Tk()
root.title("智能助手 (集成对话、图像生成和病毒扫描)")
root.geometry("1200x850")
root.protocol("WM_DELETE_WINDOW", on_closing)

# 错误映射
error_explanations = {
    400: "格式错误", 401: "认证失败", 402: "余额不足",
    422: "参数错误", 429: "请求速率达到上限",
    500: "服务器故障", 503: "服务器繁忙"
}

error_solutions = {
    400: "原因：请求体格式错误\n解决方法：请根据错误信息提示修改请求体",
    401: "原因：API key 错误，认证失败\n解决方法：请检查您的 API key 是否正确",
    402: "原因：账号余额不足\n解决方法：请确认账户余额",
    422: "原因：请求体参数错误\n解决方法：请根据错误信息提示修改相关参数",
    429: "原因：请求速率（TPM 或 RPM）达到上限\n解决方法：请合理规划您的请求速率",
    500: "原因：服务器内部故障\n解决方法：请等待后重试。若问题一直存在，请联系我们解决",
    503: "原因：服务器负载过高\n解决方法：请稍后重试您的请求"
}

# 创建主框架
main_frame = Frame(root, bg="#2d2d2d")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# 输出框框架
output_frame = Frame(main_frame, bg="#2d2d2d")
output_frame.pack(fill="both", expand=True, pady=(0, 10))

output_box = scrolledtext.ScrolledText(
    output_frame,
    wrap=tk.WORD,
    bd=0,
    highlightthickness=0,
    bg="#1e1e1e",
    fg="#ffffff",
    insertbackground="white",
    font=("Microsoft YaHei", 11)
)
output_box.pack(fill="both", expand=True, padx=5, pady=5)

# 配置标签样式
output_box.tag_config("user", foreground="#70c0ff")
output_box.tag_config("assistant", foreground="#c3e88d")
output_box.tag_config("error", foreground="#ff6666")
output_box.tag_config("warning", foreground="#ffaa00")
output_box.tag_config("success", foreground="#66cc66")

# 初始欢迎消息
output_box.insert(tk.END, "智能助手已启动！请输入您的问题开始对话。\n输入'帮助'查看使用说明\n\n")
output_box.see(tk.END)

# 图像生成面板
image_panel = Frame(main_frame, bg="#333333", padx=10, pady=5)
image_panel.pack(fill="x", padx=10, pady=5)
Label(image_panel, text="图像生成设置", bg="#333333", fg="white",
      font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")

# 尺寸选项
size_options = {
    "正方形 1:1": {"width": 1024, "height": 1024},
    "宽屏 16:9": {"width": 1344, "height": 768},
    "竖屏 9:16": {"width": 768, "height": 1344},
    "宽屏 4:3": {"width": 1024, "height": 768},
    "竖屏 3:4": {"width": 768, "height": 1024},
    "自定义尺寸": {}
}

# 尺寸选择变量
size_var = StringVar(value="正方形 1:1")
size_var.trace("w", on_size_change)

# 尺寸显示变量
size_display = StringVar(value="1024x1024")

Label(image_panel, text="图像尺寸:", bg="#333333", fg="white").grid(row=1, column=0, sticky="w")
size_menu = OptionMenu(image_panel, size_var, *size_options.keys())
size_menu.config(width=12, bg="#444444", fg="white", highlightthickness=0)
size_menu.grid(row=1, column=1, sticky="ew", padx=(0, 10))

# 尺寸显示
size_label = Label(image_panel, textvariable=size_display, bg="#333333", fg="#aaaaaa", padx=5)
size_label.grid(row=1, column=2, sticky="w")

# 自定义尺寸
Label(image_panel, text="自定义:", bg="#333333", fg="white").grid(row=2, column=0, sticky="w")
custom_width = Entry(image_panel, width=6, bg="#444444", fg="white")
custom_width.insert(0, "1024")
custom_width.grid(row=2, column=1, padx=(0, 5))
Label(image_panel, text="x", bg="#333333", f极white
").grid(row=2, column=2)
custom_height = Entry(image_panel, width=6, bg="#444444", fg="white")
custom_height.insert(0, "768")
custom_height.grid(row=2, column=3, padx=(0, 10))

# 绑定自定义尺寸变化事件
custom_width.bind("<KeyRelease>", on_custom_size_change)
custom_height.bind("<KeyRelease>", on_custom_size_change)

# 高级参数
Label(image_panel, text="迭代步数:", bg="#333333", fg="white").grid(row=3, column=0, sticky="w")
steps_var = StringVar(value="25")
steps_entry = Entry(image_panel, textvariable=steps_var, width=8, bg="#444444", fg="white")
steps_entry.grid(row=3, column=1, sticky="w")

Label(image_panel, text="指导系数:", bg="#333333", fg="white").grid(row=3, column=2, sticky="w")
guidance_var = StringVar(value="7.0")
guidance_entry = Entry(image_panel, textvariable=guidance_var, width=8, bg="#444444", fg="white")
guidance_entry.grid(row=3, column=3, sticky="w", padx=(0, 10))

Label(image_panel, text="种子:", bg="#333333", fg="white").grid(row=3, column=4, sticky="w")
seed_var = StringVar(value="-1")
seed_entry = Entry(image_panel, textvariable=seed_var, width=15, bg="#444444", fg="white")
seed_entry.grid(row=3, column=5, sticky="ew")

# 图像描述
Label(image_panel, text="图像描述:", bg="#333333", fg="white").grid(row=4, column=0, sticky="nw", pady=(10, 0))
image_prompt = tk.Text(image_panel, height=3, width=50, bg="#444444", fg="white")
image_prompt.grid(row=4, column=1, columnspan=5, pady=(10, 0), sticky="ew")

# 按钮
Button(image_panel, text="生成图像", command=generate_image_from_text,
       bg="#4a7b9d", fg="white", relief="flat", font=("Microsoft YaHei", 10)).grid(row=5, column=0, columnspan=6,
                                                                                   pady=10)

# 输入区域
input_frame = Frame(main_frame, bg="#2d2d2d")
input_frame.pack(fill="x", pady=(0, 10))

input_box = tk.Text(
    input_frame,
    height=4,
    bd=0,
    highlightthickness=0,
    bg="#3d3d3d",
    fg="white",
    insertbackground="white",
    font=("Microsoft YaHei", 11)
)
input_box.pack(fill="x", expand=True, side="left", padx=(0, 10))
input_box.focus_set()

# 按钮框架
button_frame = Frame(main_frame, bg="#2d2d2d")
button_frame.pack(fill="x", pady=(0, 10))

# 功能按钮
send_button = Button(
    button_frame, text="发送", command=send_message,
    bg="#4a7b9d", fg="white", relief="flat", width=8, font=("Microsoft YaHei", 10)
)
send_button.pack(side="left", padx=5, pady=5)

voice_button = Button(
    button_frame, text="开始语音输入", command=toggle_voice_recognition,
    bg="#5d5d5d", fg="white", relief="flat", width=12, font=("Microsoft YaHei", 10)
)
voice_button.pack(side="left", padx=5, pady=5)

image_button = Button(
    button_frame, text="隐藏图像设置", command=toggle_image_panel,
    bg="#5d5d5d", fg="white", relief="flat", width=12, font=("Microsoft YaHei", 10)
)
image_button.pack(side="left", padx=5, pady=5)

virus_button = Button(
    button_frame, text="病毒扫描", command=show_virus_scan_menu,
    bg="#9d6b4a", fg="white", relief="flat", width=10, font=("Microsoft YaHei", 10)
)
virus_button.pack(side="left", padx=5, p极=5)

help_button = Button(
    button_frame, text="帮助", command=show_help,
    bg="#5d5d5d", fg="white", relief="flat", width=8, font=("Microsoft YaHei", 10)
)
help_button.pack(side="right", padx=5, pady=5)

# 状态栏
status_frame = Frame(root, bg="#222222", height=20)
status_frame.pack(fill="x", side="bottom")

# 创建网络状态指示器
network_indicator = Frame(status_frame, width=20, height=20, bg="#222222")
network_indicator.grid(row=0, column=0, padx=(10, 0), pady=2)
network_canvas = tk.Canvas(network_indicator, width=20, height=20, bg="#222222", highlightthickness=0)
network_canvas.pack()


def update_network_status():
    """更新网络状态指示器"""
    network_canvas.delete("all")
    if is_internet_available():
        network_canvas.create_oval(2, 2, 18, 18, fill="#00ff00", outline="#00aa00")
    else:
        network_canvas.create_oval(2, 2, 18, 18, fill="#ff0000", outline="#aa0000")
    root.after(5000, update_network_status)  # 每5秒更新一次


# 状态文本
status_text = (
    f"服务状态: 对话(DeepSeek-R1) | 图像(FLUX.1) | 病毒扫描(VirusTotal) | "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
status_label = Label(
    status_frame, text=status_text,
    bg="#222222", fg="#aaaaaa", font=("Microsoft YaHei", 9)
)
status_label.grid(row=0, column=1, padx=10, sticky="w")

# 绑定快捷键
input_box.bind("<Return>", send_message)
input_box.bind("<Shift-Return>", insert_newline)
root.bind("<Control-Shift-s>", lambda e: toggle_voice_recognition())
root.bind("<Control-Shift-v>", lambda e: show_virus_scan_menu())

# ====================== 初始化系统 ======================
# 加载历史数据
load_conversation_history()

# 清理旧文件
clean_old_files()

# 初始化语音识别
voice_recognition = VoiceRecognition()

# 启动网络状态监控
update_network_status()

# 启动主循环
root.mainloop()
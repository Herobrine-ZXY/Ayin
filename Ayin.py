# 作者：PAZJ
# coding=utf-8
import base64
import hmac
import threading
import tkinter as tk
from datetime import datetime, timedelta
from hashlib import sha1
from tkinter import scrolledtext, messagebox
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets
from torchvision import transforms
import numpy as np
import requests
import json
import io
import logging
import os
import glob
import uuid
import time
from PIL import Image, ImageTk

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model.log'
)
logger = logging.getLogger(__name__)


def read_api_keys():
    try:
        with open("API.key", "r") as f:
            lines = [line.strip() for line in f.readlines()]
        if len(lines) < 1:
            raise ValueError("API.key文件至少包含1行内容")
        return {
            "deepseek_key": lines[0],

        }
    except Exception as e:
        logger.error(f"读取API密钥失败: {str(e)}")
        raise


GAN_CONFIG = {
    "image_size": (1080, 1980),
    "latent_dim": 256,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "current_epoch": 0,
    "is_training": False,
    "checkpoint_dir": "gan_models"
}


# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(GAN_CONFIG["latent_dim"], 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)


def train_gan(total_epochs):
    try:
        netG = Generator().to(GAN_CONFIG["device"])
        netD = Discriminator().to(GAN_CONFIG["device"])

        # 加载检查点
        if os.path.exists(f"{GAN_CONFIG['checkpoint_dir']}/latest.pth"):
            checkpoint = torch.load(f"{GAN_CONFIG['checkpoint_dir']}/latest.pth")
            netG.load_state_dict(checkpoint['generator'])
            netD.load_state_dict(checkpoint['discriminator'])
            GAN_CONFIG["current_epoch"] = checkpoint['epoch']

        optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        GAN_CONFIG["is_training"] = True
        dataset = datasets.FakeData(transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        dataloader = DataLoader(dataset, batch_size=GAN_CONFIG["batch_size"], shuffle=True)

        for epoch in range(GAN_CONFIG["current_epoch"], total_epochs):
            if not GAN_CONFIG["is_training"]: break

            for i, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(GAN_CONFIG["device"])
                batch_size = real_images.size(0)

                # 训练判别器
                netD.zero_grad()
                label_real = torch.full((batch_size,), 1.0, device=GAN_CONFIG["device"])
                output = netD(real_images)
                errD_real = criterion(output, label_real)

                noise = torch.randn(batch_size, GAN_CONFIG["latent_dim"], 1, 1, device=GAN_CONFIG["device"])
                fake_images = netG(noise)
                label_fake = torch.full((batch_size,), 0.0, device=GAN_CONFIG["device"])
                output = netD(fake_images.detach())
                errD_fake = criterion(output, label_fake)

                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()

                # 训练生成器
                netG.zero_grad()
                label_g = torch.full((batch_size,), 1.0, device=GAN_CONFIG["device"])
                output = netD(fake_images)
                errG = criterion(output, label_g)
                errG.backward()
                optimizerG.step()

            # 更新界面
            root.after(0, lambda: output_box.insert(tk.END,
                                                    f"Epoch [{epoch + 1}/{total_epochs}] 完成 | D Loss: {errD.item():.4f} | G Loss: {errG.item():.4f}\n"))

            # 保存模型
            if (epoch + 1) % 100 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'generator': netG.state_dict(),
                    'discriminator': netD.state_dict()
                }, f"{GAN_CONFIG['checkpoint_dir']}/latest.pth")

        GAN_CONFIG["is_training"] = False
    except Exception as e:
        logger.error(f"GAN训练错误: {str(e)}")
        root.after(0, lambda: output_box.insert(tk.END, f"训练错误: {str(e)}\n"))


def process_image_generation(prompt: str):
    try:
        netG = Generator().to(GAN_CONFIG["device"])
        checkpoint = torch.load(f"{GAN_CONFIG['checkpoint_dir']}/latest.pth")
        netG.load_state_dict(checkpoint['generator'])

        with torch.no_grad():
            noise = torch.randn(1, GAN_CONFIG["latent_dim"], 1, 1, device=GAN_CONFIG["device"])
            fake_image = netG(noise).cpu().squeeze()
            img = transforms.ToPILImage()(fake_image)
            img = img.resize((1980, 1080), Image.Resampling.LANCZOS)

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_data = img_byte_arr.getvalue()

            # 显示图像
            img_tk = ImageTk.PhotoImage(img)
            canvas.itemconfig(image_tag, image=img_tk)
            canvas.image = img_tk

            # 保存文件
            save_path = f"{IMAGE_SAVE_DIR}/gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(save_path, 'wb') as f:
                f.write(img_data)
            return f"图像已生成并保存至: {save_path}"
    except Exception as e:
        logger.error(f"图像生成失败: {str(e)}")
        return f"生成失败: {str(e)}"


# 清理20天前的日志文件
def clean_old_logs(days_to_keep=20):
    """清理指定天数前的日志文件"""
    try:
        log_files = glob.glob('model log.log')  # 获取所有日志文件

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


IMAGE_SAVE_DIR = "generated_images"  # 图片保存目录



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


def insert_newline(event=None):
    """插入换行符"""
    input_box.insert(tk.END, "\n")
    return "break"

def send_message(event=None):
    user_input = input_box.get("1.0", tk.END).strip()
    input_box.delete("1.0", tk.END)

    if user_input.startswith("GAN训练"):
        try:
            _, epoch_str = user_input.split(maxsplit=1)
            epochs = int(epoch_str)
            if epochs <= 0:
                raise ValueError
            if GAN_CONFIG["is_training"]:
                safe_update_output("训练已在进行中")
                return
            threading.Thread(target=train_gan, args=(epochs,), daemon=True).start()
            safe_update_output(f"开始GAN训练，总epoch数: {epochs}")
        except Exception as e:
            safe_update_output("命令格式错误，正确示例：GAN训练 100")
    elif user_input == "停止GAN训练":
        if GAN_CONFIG["is_training"]:
            GAN_CONFIG["is_training"] = False
            safe_update_output("已发送停止指令")
        else:
            safe_update_output("当前未在训练")
    elif any(cmd in user_input.lower() for cmd in ["生成图", "生成图像", "生成图片"]):
        prompt = user_input.split(maxsplit=1)[1].strip() if len(user_input.split()) > 1 else ""
        if not prompt:
            safe_update_output("请提供图像描述")
            return
        threading.Thread(target=lambda: process_image_generation(prompt)).start()
    else:
        # 原有API调用逻辑
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                timeout=30
            )
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
                safe_update_output(f"错误：{str(e)}")
            logger.error(f"获取助手回复失败: {str(e)}")

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
    root, wrap=tk.WORD, width=100, height=30, bd=0,
    highlightthickness=0, bg="#2d2d2d", fg="white",
    insertbackground="white"
)
canvas.create_window(100, 100, anchor="nw", window=output_box, width=800, height=600)

input_box = tk.Text(
    root, height=5, width=100, bd=0, highlightthickness=0,
    bg="#2d2d2d", fg="white", insertbackground="white"
)
canvas.create_window(100, 720, anchor="nw", window=input_box, width=800, height=150)

# 图像显示区域
image_tag = canvas.create_image(1200, 200, anchor="nw")

# 绑定按键事件
input_box.bind("<Return>", send_message)
input_box.bind("<Shift-Return>", insert_newline)

# 运行主循环
root.mainloop()
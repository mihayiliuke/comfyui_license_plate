import numpy as np
import torch
import os
import io
import json
import torchaudio
import folder_paths
import cv2
import numpy as np
from pathlib import Path
import folder_paths

import cv2
import numpy as np
import torch
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

class LicensePlateReplaceOptimizedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg_image": ("IMAGE",),
                "plate_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace"
    CATEGORY = "custom/plate"
    
    def tensor_to_np(self, tensor):
        return (tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    def np_to_tensor(self, image):
        return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    def replace_plate_in_image(self, orig, seg, plate, scale_factor=4, expand_ratio=0.04, min_contour_area=1000):
        def sample_contour_points(contour, num_points=200):
            contour = contour[:, 0]
            n = len(contour)
            segment_lengths = np.linalg.norm(contour[(np.arange(n) + 1) % n] - contour, axis=1)
            total_length = np.sum(segment_lengths)
            cum_lengths = np.cumsum(segment_lengths)

            sample_locs = np.random.uniform(0, total_length, num_points)
            sample_locs.sort()

            points = []
            seg_idx = 0
            curr_len = 0.0
            for loc in sample_locs:
                while seg_idx < len(segment_lengths) and curr_len + segment_lengths[seg_idx] < loc:
                    curr_len += segment_lengths[seg_idx]
                    seg_idx += 1
                if seg_idx >= len(segment_lengths):
                    seg_idx = len(segment_lengths) - 1

                a = contour[seg_idx]
                b = contour[(seg_idx + 1) % n]
                t = (loc - curr_len) / segment_lengths[seg_idx]
                pt = (1 - t) * a + t * b
                points.append(pt)
            return np.array(points, dtype=np.float32)
        
        def point_to_segment_dist(points, a, b):
            ab = b - a
            ap = points - a
            ab_len_sq = np.sum(ab ** 2)
            t = np.clip(np.sum(ap * ab, axis=1) / (ab_len_sq + 1e-6), 0, 1).reshape(-1, 1)
            proj = a + t * ab
            return np.linalg.norm(points - proj, axis=1)

        def compute_loss(points, quad):
            total = np.zeros(len(points))
            for i in range(4):
                a = quad[i]
                b = quad[(i + 1) % 4]
                d = point_to_segment_dist(points, a, b)
                total = np.minimum(total, d) if i > 0 else d
            return total.sum()

        def optimize_quad(points, initial_quad, iterations=1000, lr=0.001):
            quad = initial_quad.astype(np.float32)
            for _ in range(iterations):
                grad = np.zeros_like(quad)
                loss0 = compute_loss(points, quad)
                eps = 1e-2
                for i in range(4):
                    for j in range(2):
                        quad[i, j] += eps
                        loss1 = compute_loss(points, quad)
                        grad[i, j] = (loss1 - loss0) / eps
                        quad[i, j] -= eps
                quad -= lr * grad
            return quad

        def get_quad_covering_contour(cnt):
            initial_quad = cv2.boxPoints(cv2.minAreaRect(cnt))
            sampled_points = sample_contour_points(cnt, num_points=1000)
            optimized_quad = optimize_quad(sampled_points, initial_quad)
            return optimized_quad.astype('float32')

        def sort_pts(pts):
            pts = np.array(pts, dtype='float32').reshape(4, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            return np.array([tl, tr, br, bl], dtype='float32')

        orig_high = cv2.resize(orig, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        seg_high = cv2.resize(seg, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(seg_high, cv2.COLOR_BGR2GRAY)
        _, mask_src = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_src = cv2.GaussianBlur(mask_src, (5, 5), 0)
        _, mask_src = cv2.threshold(mask_src, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('未检测到车牌区域')
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_contour_area:
            raise ValueError('检测到的车牌轮廓面积过小')

        quad_pts = get_quad_covering_contour(cnt)
        src_quad = sort_pts(quad_pts.astype(np.float32))

        center = np.mean(src_quad, axis=0)
        src_quad -= center
        src_quad *= (1 + expand_ratio)
        src_quad += center

        box_width = np.max(src_quad[:, 0]) - np.min(src_quad[:, 0])
        box_height = np.max(src_quad[:, 1]) - np.min(src_quad[:, 1])
        scale = min(box_width / plate.shape[1], box_height / plate.shape[0]) * 0.95
        new_size = (int(plate.shape[1] * scale * scale_factor), int(plate.shape[0] * scale * scale_factor))
        plate_high = cv2.resize(plate, new_size, interpolation=cv2.INTER_CUBIC)

        h_p, w_p = plate_high.shape[:2]
        dst_quad = np.array([[0, 0], [w_p-1, 0], [w_p-1, h_p-1], [0, h_p-1]], dtype='float32')
        M = cv2.getPerspectiveTransform(dst_quad, src_quad)
        warped_plate = cv2.warpPerspective(plate_high, M, (orig_high.shape[1], orig_high.shape[0]), flags=cv2.INTER_CUBIC)

        result_high = orig_high.copy()
        gray_plate = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
        _, plate_mask = cv2.threshold(gray_plate, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(plate_mask)

        for c in range(3):
            result_high[:, :, c] = cv2.bitwise_and(result_high[:, :, c], mask_inv)
        cv2.copyTo(warped_plate, plate_mask, result_high)

        result = cv2.resize(result_high, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
        return result
    def replace(self, image, seg_image, plate_image):
        def tensor_to_bgr(img_tensor):
            arr = img_tensor.cpu().numpy()
            if arr.ndim == 4:
                arr = arr[0]
                if arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            if arr.shape[2] == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return arr
        def bgr_to_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            tensor = tensor.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
            print(f"Converted BGR to tensor with shape: {tensor.shape}")
            return tensor
        orig = tensor_to_bgr(image)
        seg = tensor_to_bgr(seg_image)
        plate = tensor_to_bgr(plate_image)

        print(f"Original shape: {orig.shape}, Segmentation shape: {seg.shape}, Plate shape: {plate.shape}")

        result_np = self.replace_plate_in_image(orig, seg, plate)
        result_tensor = bgr_to_tensor(result_np)
        return (result_tensor,)

class AudioOverlayNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
            },
            "optional": {
                "volume1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "volume2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "音频处理"
    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "overlay_and_save"
    OUTPUT_NODE = True

    def overlay_and_save(self, audio1, audio2, filename_prefix="audio/ComfyUI", volume1=1.0, volume2=1.0, prompt=None, extra_pnginfo=None):
        if audio1 is None or audio2 is None:
            raise ValueError("音频输入不能为 None")
        
        data1 = audio1["waveform"]
        sr1 = audio1["sample_rate"]
        data2 = audio2["waveform"]
        sr2 = audio2["sample_rate"]

        data1 = data1.squeeze(0).numpy()
        data2 = data2.squeeze(0).numpy()

        if sr1 != sr2:
            raise ValueError("两段音频的采样率必须一致！")

        max_length = max(data1.shape[1], data2.shape[1])
        data1_padded = np.pad(data1, ((0, 0), (0, max_length - data1.shape[1])), mode='constant')
        data2_padded = np.pad(data2, ((0, 0), (0, max_length - data2.shape[1])), mode='constant')

        data1_adjusted = data1_padded * volume1
        data2_adjusted = data2_padded * volume2
        mixed_data = data1_adjusted + data2_adjusted

        max_amplitude = np.max(np.abs(mixed_data))
        if max_amplitude > 1.0:
            mixed_data = mixed_data / max_amplitude

        waveform = torch.from_numpy(mixed_data).to(torch.float32)

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        file = f"{filename}_{counter:05}_.wav"  # ✅ 修改为 WAV 文件名

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

        # ✅ 直接保存为 WAV（无需 BytesIO 缓冲区）
        os.makedirs(full_output_folder, exist_ok=True)
        torchaudio.save(os.path.join(full_output_folder, file), waveform, sr1)  # ✅ 保存为 wav（默认格式）

        result = {
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }
        print(f"Saved audio: {result}")

        # ✅ 模仿 TangoFlux 返回结构
        audio_for_vhs = {
            "waveform": waveform.unsqueeze(0),  # 保持格式为 (1, 2, N)
            "sample_rate": sr1
        }
        print("mikhail")
        print(f'[result] is {[result]}')
        print(f'audio_for_vhs is {audio_for_vhs}')
        #print(audio_for_vhs,)
        return {
            "ui": {"audios": [result]},
            "result": (audio_for_vhs,)
        }

    def insert_or_replace_vorbis_comment(self, buff, metadata):
        return buff

class AudioTrimNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.1}),
            }
        }

    CATEGORY = "音频处理"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("trimmed_audio",)
    FUNCTION = "trim_audio"

    def trim_audio(self, audio, start_time=0.0, end_time=5.0):
        waveform = audio["waveform"]  # shape: (1, channels, samples)
        sr = audio["sample_rate"]
        channels, samples = waveform.shape[1], waveform.shape[2]

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 限制索引范围
        start_sample = max(0, min(start_sample, samples))
        end_sample = max(start_sample, min(end_sample, samples))

        trimmed = waveform[:, :, start_sample:end_sample]  # 保持原格式
        return ({"waveform": trimmed, "sample_rate": sr},)


import torch
import numpy as np
import folder_paths

class AudioLoudnessNormalizeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_loudness": ("FLOAT", {
                    "default": -20.0,
                    "min": -60.0,
                    "max": 0.0,
                    "step": 1.0
                }),
            }
        }

    CATEGORY = "音频处理"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("normalized_audio",)
    FUNCTION = "normalize_loudness"

    def normalize_loudness(self, audio, target_loudness=-20.0):
        waveform = audio["waveform"]  # shape: (1, channels, samples)
        sr = audio["sample_rate"]

        waveform_np = waveform.squeeze(0).numpy()  # shape: (channels, samples)

        # 估算当前响度（使用最大幅值）
        peak_amplitude = np.max(np.abs(waveform_np))
        if peak_amplitude == 0:
            gain = 1.0  # 避免 log(0)
        else:
            current_loudness = 20 * np.log10(peak_amplitude)
            gain = 10 ** ((target_loudness - current_loudness) / 20)

        # 应用增益
        adjusted_np = waveform_np * gain

        # 防止 clipping（超过 [-1, 1]）
        max_val = np.max(np.abs(adjusted_np))
        if max_val > 1.0:
            adjusted_np = adjusted_np / max_val

        # 转换回 tensor 格式
        adjusted_tensor = torch.from_numpy(adjusted_np).unsqueeze(0).to(torch.float32)

        return ({"waveform": adjusted_tensor, "sample_rate": sr},)

import torch
import numpy as np

class AudioNormalizeLoudnessNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "min_db": ("FLOAT", {"default": -35.0, "min": -60.0, "max": 0.0, "step": 1.0}),
                "max_db": ("FLOAT", {"default": -5.0, "min": -60.0, "max": 0.0, "step": 1.0}),
            }
        }

    CATEGORY = "音频处理"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("normalized_audio",)
    FUNCTION = "normalize_audio"

    def normalize_audio(self, audio, min_db=-35.0, max_db=-5.0):
        waveform = audio["waveform"]  # shape: (1, 2, N)
        sr = audio["sample_rate"]
        data = waveform.squeeze(0).numpy()  # shape: (2, N)

        # 计算每通道响度 (dBFS)
        rms = np.sqrt(np.mean(data**2, axis=1)) + 1e-9  # 防止 log(0)
        dbfs = 20 * np.log10(rms)

        # 目标范围中心 dB
        target_db = (min_db + max_db) / 2

        # 计算增益因子（每个通道）
        gain_db = target_db - dbfs
        gain = 10 ** (gain_db[:, np.newaxis] / 20.0)
        adjusted_data = data * gain

        # 限幅防止削波
        max_val = np.max(np.abs(adjusted_data))
        if max_val > 1.0:
            adjusted_data = adjusted_data / max_val

        normalized_waveform = torch.from_numpy(adjusted_data).unsqueeze(0).to(torch.float32)  # shape: (1, 2, N)
        return ({"waveform": normalized_waveform, "sample_rate": sr},)

import torch
import torchaudio
import numpy as np
import folder_paths
import os

class AudioCompressorNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0, "step": 1.0}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "attack": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001}),
                "release": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01}),
                "makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "compress_audio"
    CATEGORY = "音频处理"

    def compress_audio(self, audio, threshold_db, ratio, attack, release, makeup_gain_db):
        waveform = audio["waveform"].squeeze(0)  # (channels, samples)
        sample_rate = audio["sample_rate"]
        channels, samples = waveform.shape

        # 窗口设置（10ms 一帧）
        frame_size = int(sample_rate * 0.01)
        envelope = torch.zeros_like(waveform)
        gain = torch.ones_like(waveform)

        for c in range(channels):
            prev_gain = 1.0
            for i in range(0, samples, frame_size):
                frame = waveform[c, i:i+frame_size]
                if frame.numel() == 0:
                    continue
                rms = torch.sqrt(torch.mean(frame ** 2))
                if rms.item() < 1e-8:
                    rms_db = -100.0
                else:
                    rms_db = 20 * torch.log10(rms)

                over_threshold = rms_db - threshold_db
                if over_threshold <= 0:
                    desired_gain_db = 0
                else:
                    desired_gain_db = -over_threshold * (1.0 - 1.0 / ratio)

                desired_gain = 10 ** (desired_gain_db / 20)
                current_gain = prev_gain

                # 平滑处理：attack / release
                for j in range(frame.numel()):
                    if desired_gain < current_gain:
                        coeff = 1 - np.exp(-1.0 / (sample_rate * attack))
                    else:
                        coeff = 1 - np.exp(-1.0 / (sample_rate * release))
                    current_gain += (desired_gain - current_gain) * coeff
                    idx = i + j
                    if idx < samples:
                        gain[c, idx] = current_gain
                prev_gain = current_gain

        # 应用压缩增益
        compressed_waveform = waveform * gain

        # 增益补偿
        makeup_gain = 10 ** (makeup_gain_db / 20)
        compressed_waveform *= makeup_gain

        # 避免爆音（归一化）
        max_amp = compressed_waveform.abs().max()
        if max_amp > 1.0:
            compressed_waveform /= max_amp

        return ({
            "waveform": compressed_waveform.unsqueeze(0),  # (1, C, N)
            "sample_rate": sample_rate
        },)

import torch
import folder_paths

class AudioFadeNode:
    def __init__(self):
        self.type = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "fade_out_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_fade"
    CATEGORY = "音频处理"

    def apply_fade(self, audio, fade_in_duration, fade_out_duration):
        waveform = audio["waveform"].squeeze(0)  # (channels, samples)
        sample_rate = audio["sample_rate"]
        channels, samples = waveform.shape

        fade_in_samples = int(fade_in_duration * sample_rate)
        fade_out_samples = int(fade_out_duration * sample_rate)

        envelope = torch.ones(samples, dtype=waveform.dtype)

        # 淡入：线性增长
        if fade_in_samples > 0:
            fade_in_curve = torch.linspace(0.0, 1.0, steps=fade_in_samples)
            envelope[:fade_in_samples] = fade_in_curve

        # 淡出：线性下降
        if fade_out_samples > 0:
            fade_out_curve = torch.linspace(1.0, 0.0, steps=fade_out_samples)
            envelope[-fade_out_samples:] = envelope[-fade_out_samples:] * fade_out_curve

        # 扩展至多通道
        envelope = envelope.unsqueeze(0).expand(channels, -1)
        faded_waveform = waveform * envelope

        return ({
            "waveform": faded_waveform.unsqueeze(0),  # (1, C, N)
            "sample_rate": sample_rate
        },)
import torch
import torchaudio
import os
import folder_paths

class AudioFormatConverterNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_format": (["wav", "mp3", "flac"], {"default": "wav"}),
                "bit_depth": (["16", "24", "32"], {"default": "16"}),
                "sample_rate": ("INT", {"default": 48000, "min": 8000, "max": 192000, "step": 1000}),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_converted"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "音频处理"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_format"
    OUTPUT_NODE = True

    def convert_format(self, audio, output_format="wav", bit_depth="16", sample_rate=48000, filename_prefix="audio/ComfyUI_converted", prompt=None, extra_pnginfo=None):
        if audio is None:
            raise ValueError("音频输入不能为 None")

        # 获取输入音频数据
        waveform = audio["waveform"]  # shape: (1, channels, samples)
        input_sr = audio["sample_rate"]

        # 转换为 numpy 并去掉 batch 维度
        waveform_np = waveform.squeeze(0).numpy()  # shape: (channels, samples)

        # 重采样（如果需要）
        if input_sr != sample_rate:
            waveform_tensor = torch.from_numpy(waveform_np).to(torch.float32)
            waveform_tensor = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=sample_rate)(waveform_tensor)
            waveform_np = waveform_tensor.numpy()

        # 转换为 torch tensor 准备保存
        waveform_tensor = torch.from_numpy(waveform_np).to(torch.float32)

        # 设置比特深度
        if bit_depth == "16":
            subtype = "PCM_16"
        elif bit_depth == "24":
            subtype = "PCM_24"
        else:  # 32
            subtype = "PCM_32"

        # 文件名处理
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.{output_format}"

        # 保存元数据
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

        # 创建输出目录并保存音频
        os.makedirs(full_output_folder, exist_ok=True)
        output_path = os.path.join(full_output_folder, file)
        
        # 保存音频文件
        if output_format == "mp3":
            # MP3 需要特别处理比特率而不是比特深度
            torchaudio.save(output_path, waveform_tensor, sample_rate, format="mp3", bits_per_sample=16)  # MP3 通常使用 16-bit
        else:
            torchaudio.save(output_path, waveform_tensor, sample_rate, format=output_format, bits_per_sample=int(bit_depth))

        # 返回结果
        result = {
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }
        print(f"Saved audio: {result}")

        # 返回 VHS 兼容的音频格式
        audio_for_vhs = {
            "waveform": waveform_tensor.unsqueeze(0),  # shape: (1, channels, samples)
            "sample_rate": sample_rate
        }

        return {
            "ui": {"audios": [result]},
            "result": (audio_for_vhs,)
        }

NODE_CLASS_MAPPINGS = {
    "AudioOverlay": AudioOverlayNode,
    "AudioTrim": AudioTrimNode,
    "AudioLoudnessNormalize": AudioLoudnessNormalizeNode,
    "AudioNormalizeLoudness": AudioNormalizeLoudnessNode,
    "AudioCompressor": AudioCompressorNode,
    "AudioFade": AudioFadeNode,
    "AudioFormatConverter":AudioFormatConverterNode,
    "LicensePlateReplaceSimpleNode": LicensePlateReplaceOptimizedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioOverlay": "叠加两段音频并保存",
    "AudioTrim": "裁剪音频（按时间）",
    "AudioLoudnessNormalize": "压缩音频到目标响度",
    "AudioNormalizeLoudness": "响度压缩（标准化到 dB）",
    "AudioCompressor": "音频压缩器（Compressor）",
    "AudioFade": "音频淡入淡出器（Fade In/Out）",
    "AudioFormatConverter":"音频格式转换器（Format Converter）",
    "LicensePlateReplaceSimpleNode": "车牌透视替换节点"
}

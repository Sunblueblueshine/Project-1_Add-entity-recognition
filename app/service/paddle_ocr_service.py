import io
import logging
import os
import fitz  # pymupdf
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any
from app.config.paddle_ocr_config import default_paddle_ocr_config

logger = logging.getLogger(__name__)

class PaddleOCRService:
    """PaddleOCR文本识别服务"""
    
    def __init__(self):
        self._is_initialized = False
        self._ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化PaddleOCR引擎"""
        try:
            logger.info("开始初始化PaddleOCR引擎...")
            
            # 创建PaddleOCR实例，使用您配置的模型
            self._ocr = PaddleOCR(
                text_detection_model_name=default_paddle_ocr_config.TEXT_DETECTION_MODEL_NAME,
                text_recognition_model_name=default_paddle_ocr_config.TEXT_RECOGNITION_MODEL_NAME,
                use_doc_orientation_classify=default_paddle_ocr_config.USE_DOC_ORIENTATION_CLASSIFY,
                use_doc_unwarping=default_paddle_ocr_config.USE_DOC_UNWARPING,
                use_textline_orientation=default_paddle_ocr_config.USE_TEXTLINE_ORIENTATION
            )
            
            logger.info("PaddleOCR引擎初始化成功")
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"初始化PaddleOCR引擎失败: {str(e)}")
            self._is_initialized = False
    
    def is_available(self) -> bool:
        """检查OCR服务是否可用"""
        logger.debug("检查PaddleOCR服务可用性")
        if not self._is_initialized:
            logger.debug("PaddleOCR服务未初始化，尝试初始化...")
            self._initialize_ocr()
        status = "可用" if self._is_initialized else "不可用"
        logger.debug(f"PaddleOCR服务状态: {status}")
        return self._is_initialized
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> np.ndarray:
        """使用pymupdf将PDF页面转换为图像，支持大尺寸图像智能缩放"""
        logger.info(f"开始将PDF页面转换为图像: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # 检查页面索引是否有效
            if page_num < 1 or page_num > total_pages:
                raise IndexError(f"页面索引超出范围: {page_num}，总页数: {total_pages}")
            
            page = doc[page_num - 1]  # pymupdf的页面索引从0开始
            
            # 获取页面原始尺寸
            page_rect = page.rect
            original_width = page_rect.width
            original_height = page_rect.height
            
            # 计算合适的DPI，确保图像尺寸不超过限制
            max_width = default_paddle_ocr_config.MAX_IMAGE_WIDTH
            max_height = default_paddle_ocr_config.MAX_IMAGE_HEIGHT
            
            # 计算缩放比例
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y, 1.0)  # 不超过原始尺寸
            
            # 计算实际DPI
            actual_dpi = default_paddle_ocr_config.DPI * scale
            zoom = actual_dpi / 72  # 72是PDF的默认DPI
            
            logger.debug(f"图像尺寸控制: 原始尺寸={original_width:.0f}x{original_height:.0f}, "
                        f"最大限制={max_width}x{max_height}, 缩放比例={scale:.3f}, "
                        f"实际DPI={actual_dpi:.0f}")
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为PIL图像
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # 检查最终图像尺寸
            final_width, final_height = img.size
            final_pixels = final_width * final_height
            
            if final_pixels > default_paddle_ocr_config.MAX_PIXELS:
                logger.warning(f"图像尺寸仍然过大: {final_width}x{final_height} ({final_pixels}像素), "
                             f"最大限制: {default_paddle_ocr_config.MAX_PIXELS}像素")
                # 进一步缩放
                scale_factor = (default_paddle_ocr_config.MAX_PIXELS / final_pixels) ** 0.5
                new_width = int(final_width * scale_factor)
                new_height = int(final_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"图像已进一步缩放至: {new_width}x{new_height}")
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            doc.close()
            logger.info(f"PDF页面转换为图像成功: 文件={os.path.basename(pdf_path)}, 页码={page_num}, "
                       f"最终图像尺寸={img_array.shape}")
            return img_array
        except Exception as e:
            logger.error(f"PDF页面转换为图像失败: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 错误={str(e)}")
            raise
    
    
    def _process_paddle_ocr_result(self, ocr_result) -> str:
        """处理PaddleOCR识别结果，提取文本内容"""
        try:
            if not ocr_result:
                logger.debug("OCR结果为空")
                return ""
            
            logger.debug(f"OCR结果类型: {type(ocr_result)}")
            
            # 从PaddleOCR结果中提取文本
            text_lines = []
            
            # 新的PaddleOCR返回格式：OCRResult对象，包含json属性
            if ocr_result and len(ocr_result) > 0:
                ocr_result_obj = ocr_result[0]
                
                # 检查是否有json属性
                if hasattr(ocr_result_obj, 'json'):
                    json_data = ocr_result_obj.json
                    logger.debug(f"JSON数据键: {list(json_data.keys())}")
                    
                    # 检查是否有res字段（新的数据格式）
                    if 'res' in json_data:
                        res_data = json_data['res']
                        logger.debug(f"res数据类型: {type(res_data)}")
                        
                        # 从res数据中提取文本
                        if isinstance(res_data, dict) and 'rec_texts' in res_data and res_data['rec_texts']:
                            rec_texts = res_data['rec_texts']
                            rec_scores = res_data.get('rec_scores', [])
                            
                            logger.debug(f"res中rec_texts数量: {len(rec_texts)}")
                            logger.debug(f"res中rec_scores数量: {len(rec_scores)}")
                            
                            # 确保文本和分数数量一致
                            min_length = min(len(rec_texts), len(rec_scores)) if rec_scores else len(rec_texts)
                            
                            for i in range(min_length):
                                text_content = rec_texts[i] if i < len(rec_texts) else ""
                                entities = default_entity_rec_service.recognize(text_content)
                                confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                                
                                logger.debug(f"识别文本: '{text_content}', 置信度: {confidence:.3f}")
                                
                                # 只保留置信度高于阈值且非空的文本
                                if confidence >= default_paddle_ocr_config.MIN_CONFIDENCE and text_content.strip():
                                    text_lines.append(text_content.strip())
                                    logger.debug(f"✅ 保留文本: '{text_content}'")
                                else:
                                    logger.debug(f"❌ 丢弃文本: '{text_content}' (置信度: {confidence:.3f})")
                        else:
                            logger.debug("res数据中没有rec_texts或为空")
                    
                    # 兼容旧格式：直接从JSON数据中提取文本
                    elif 'rec_texts' in json_data and json_data['rec_texts']:
                        rec_texts = json_data['rec_texts']
                        rec_scores = json_data.get('rec_scores', [])
                        
                        logger.debug(f"JSON中rec_texts数量: {len(rec_texts)}")
                        logger.debug(f"JSON中rec_scores数量: {len(rec_scores)}")
                        
                        # 确保文本和分数数量一致
                        min_length = min(len(rec_texts), len(rec_scores)) if rec_scores else len(rec_texts)
                        
                        for i in range(min_length):
                            text_content = rec_texts[i] if i < len(rec_texts) else ""
                            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                            
                            logger.debug(f"识别文本: '{text_content}', 置信度: {confidence:.3f}")
                            
                            # 只保留置信度高于阈值且非空的文本
                            if confidence >= default_paddle_ocr_config.MIN_CONFIDENCE and text_content.strip():
                                text_lines.append(text_content.strip())
                                logger.debug(f"✅ 保留文本: '{text_content}'")
                            else:
                                logger.debug(f"❌ 丢弃文本: '{text_content}' (置信度: {confidence:.3f})")
                    else:
                        logger.debug("JSON数据中没有rec_texts或为空")
            
            # 合并所有文本行，保留换行，仅折叠行内多空格
            logger.debug(f"提取到文本行数: {len(text_lines)}")
            if text_lines:
                import re
                normalized_lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text_lines if ln and ln.strip()]
                full_text = '\n'.join(normalized_lines).strip()
                logger.debug(f"合并后文本长度: {len(full_text)}")
            else:
                full_text = ""

            return full_text
            
        except Exception as e:
            logger.error(f"处理PaddleOCR识别结果失败: {str(e)}")
            return ""
    
    def recognize_text(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """识别PDF页面中的文本，使用PaddleOCR，支持内存优化和错误恢复"""
        try:
            logger.info(f"开始OCR文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            # 尝试获取图像，如果失败则使用降级策略
            try:
                img_array = self._pdf_page_to_image(pdf_path, page_num)
            except Exception as img_error:
                logger.warning(f"图像转换失败，尝试降级处理: {str(img_error)}")
                # 使用更保守的设置重试
                img_array = self._pdf_page_to_image_fallback(pdf_path, page_num)
            
            # 调用PaddleOCR进行文本识别
            logger.info(f"调用PaddleOCR进行文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            try:
                ocr_result = self._ocr.ocr(img_array)
            except Exception as ocr_error:
                logger.error(f"PaddleOCR处理失败: {str(ocr_error)}")
                # 如果是内存错误，尝试进一步缩小图像
                if "allocate" in str(ocr_error).lower() or "memory" in str(ocr_error).lower():
                    logger.info("检测到内存错误，尝试使用更小的图像重试")
                    img_array = self._pdf_page_to_image_fallback(pdf_path, page_num, aggressive=True)
                    ocr_result = self._ocr.ocr(img_array)
                else:
                    raise ocr_error
            
            # 处理识别结果
            text = self._process_paddle_ocr_result(ocr_result)
            
            logger.debug(f"PaddleOCR识别完成，获取到文本")
            
            # 处理识别结果
            text_lines_count = len(text.strip().split('\n')) if text else 0
            
            recognized_text_length = len(text.strip())
            logger.info(f"OCR文本识别完成: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 识别文本长度={recognized_text_length}, 识别文本行数={text_lines_count}")
            
            return {
                'text': text.strip(),
                'tables': []  # 不再识别表格，返回空列表
            }
            
        except Exception as e:
            logger.error(f"OCR识别失败 (页面 {page_num}): {str(e)}")
            return {
                'text': "",
                'tables': []
            }
    
    def _pdf_page_to_image_fallback(self, pdf_path: str, page_num: int, aggressive: bool = False) -> np.ndarray:
        """降级图像转换方法，使用更保守的设置"""
        logger.info(f"使用降级图像转换: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 激进模式={aggressive}")
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            
            # 使用更保守的DPI设置
            if aggressive:
                fallback_dpi = 150  # 更低的DPI
                max_size = 2000  # 更小的最大尺寸
            else:
                fallback_dpi = 200
                max_size = 2500
            
            # 获取页面尺寸并计算缩放
            page_rect = page.rect
            original_width = page_rect.width
            original_height = page_rect.height
            
            scale_x = max_size / original_width
            scale_y = max_size / original_height
            scale = min(scale_x, scale_y, 1.0)
            
            actual_dpi = fallback_dpi * scale
            zoom = actual_dpi / 72
            
            logger.debug(f"降级转换: 原始={original_width:.0f}x{original_height:.0f}, "
                        f"缩放={scale:.3f}, DPI={actual_dpi:.0f}")
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # 确保图像不会太大
            final_width, final_height = img.size
            if final_width * final_height > max_size * max_size:
                scale_factor = (max_size * max_size / (final_width * final_height)) ** 0.5
                new_width = int(final_width * scale_factor)
                new_height = int(final_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"降级图像进一步缩放至: {new_width}x{new_height}")
            
            img_array = np.array(img)
            doc.close()
            
            logger.info(f"降级图像转换成功: 最终尺寸={img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"降级图像转换失败: {str(e)}")
            raise
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """直接识别图像文件中的文本"""
        try:
            logger.info(f"开始OCR图像识别: {os.path.basename(image_path)}")
            
            # 读取图像文件
            from PIL import Image
            import numpy as np
            
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # 调用PaddleOCR进行文本识别
            logger.info(f"调用PaddleOCR进行图像识别: {os.path.basename(image_path)}")
            
            ocr_result = self._ocr.ocr(img_array)
            
            # 处理识别结果
            text = self._process_paddle_ocr_result(ocr_result)
            
            logger.debug(f"PaddleOCR图像识别完成，获取到文本")
            
            # 处理识别结果
            text_lines_count = len(text.strip().split('\n')) if text else 0
            
            recognized_text_length = len(text.strip())
            logger.info(f"OCR图像识别完成: 文件={os.path.basename(image_path)}, 识别文本长度={recognized_text_length}, 识别文本行数={text_lines_count}")
            
            return {
                'text': text.strip(),
                'tables': []  # 不再识别表格，返回空列表
            }
            
        except Exception as e:
            logger.error(f"OCR图像识别失败: {str(e)}")
            return {
                'text': "",
                'tables': []
            }
    
    def process_page_with_ocr_fallback(self, pdf_path: str, page_num: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """当pdfplumber提取的文本不足时，使用OCR作为后备方案"""
        current_text_length = len(page_data.get('text', '').strip())
        
        if current_text_length < default_paddle_ocr_config.OCR_THRESHOLD:
            logger.info(f"页面 {page_num} 文本长度不足，触发OCR识别")
            ocr_result = self.recognize_text(pdf_path, page_num)
            
            if ocr_result.get('text'):
                page_data['text'] = ocr_result['text']
                logger.info(f"OCR识别成功，文本长度: {len(ocr_result['text'])}")
        
        return page_data

# 创建全局OCR服务实例
ocr_service = PaddleOCRService()

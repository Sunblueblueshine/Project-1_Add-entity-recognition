"""
文本处理工具模块
"""
import re
import docx
import pdfplumber
import logging
from typing import List, Dict, Any, Optional
from app.config.similarity_config import default_config
from app.config.synonyms_config import SYNONYMS
from app.service.paddle_ocr_service import ocr_service as paddle_ocr_service

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str, page_range: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """从PDF文件中按页提取文本和表格 - 全用OCR提取"""
    pages = []
    
    # 检查OCR服务是否可用
    if not paddle_ocr_service.is_available():
        logger.warning("OCR服务不可用，回退到pdfplumber提取")
        return _extract_with_pdfplumber(pdf_path)
    
    try:
        # 首先获取PDF总页数
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"开始使用OCR提取PDF文本，总页数: {total_pages}")
        
        # 确定要处理的页面范围
        if page_range:
            pages_to_process = [p for p in page_range if 1 <= p <= total_pages]
            logger.info(f"指定页面范围: {page_range}, 有效页面: {pages_to_process}")
        else:
            pages_to_process = list(range(1, total_pages + 1))
            logger.info(f"处理所有页面: 1-{total_pages}")
        
        # 逐页使用OCR提取
        for page_num in pages_to_process:
            try:
                logger.info(f"开始OCR提取第{page_num}页，总页数: {total_pages}")
                ocr_result = paddle_ocr_service.recognize_text(pdf_path, page_num)
                
                text_content = ocr_result.get('text', '')
                logger.info(f"OCR提取第{page_num}页完成，文本长度: {len(text_content)}")
                
                # 按页面格式提取，每页作为一个独立的文本段落
                if text_content.strip():  # 只添加非空页面
                    pages.append({
                        'page_num': page_num,
                        'text': text_content,
                        'tables': ocr_result.get('tables', [])
                    })
                else:
                    logger.debug(f"第{page_num}页文本为空，跳过")
                
            except IndexError as e:
                logger.error(f"页面索引错误: {str(e)}")
                # 页面索引错误，跳过该页面
                continue
            except Exception as e:
                logger.error(f"OCR提取第{page_num}页失败: {str(e)}")
                # 如果单页OCR失败，不添加空页面，直接跳过
                continue
            
    except Exception as e:
        logger.error(f"OCR提取失败: {str(e)}，回退到pdfplumber")
        return _extract_with_pdfplumber(pdf_path)
    
    # 检查提取结果
    if not pages:
        logger.warning("OCR提取结果为空，尝试使用pdfplumber")
        return _extract_with_pdfplumber(pdf_path)
    
    # 统计提取结果
    total_text_length = sum(len(page.get('text', '')) for page in pages)
    non_empty_pages = len(pages)  # 现在pages中只包含非空页面
    
    logger.info(f"OCR提取完成: 有效页数={len(pages)}, 总文本长度={total_text_length}")
    
    return pages

def _extract_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """使用pdfplumber作为后备提取方法（简化版，不处理表格）"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_data = {
                'page_num': page.page_number,
                'text': page.extract_text() or "",
                'tables': []  # OCR模式下不处理表格
            }
            pages.append(page_data)
    return pages

def extract_kv_tables_from_text(text: str) -> List[Dict[str, Any]]:
    """基于规则从页面文本中抽取近似表格的KV块（最小可用版本）。

    规则：
    - 按行遍历，识别两列样式：
      1) key: value 或 key：value
      2) key  [2个及以上空格]  value
    - 连续满足规则的行数 >= TABLE_MIN_ROWS 视为一个表格块。
    - 返回items（dict）与raw_rows（原始行）。同key保留首次出现的值。
    """
    min_rows = getattr(default_config, 'TABLE_MIN_ROWS', 3)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    blocks: List[Dict[str, Any]] = []

    def parse_kv(line: str) -> Optional[tuple]:
        # 模式1：冒号
        m = re.match(r"^(.{1,40})[：:]\s*(.+)$", line)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # 模式2：多空格分列
        m2 = re.match(r"^(.{1,40}?)[\t ]{2,}(.+)$", line)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()
        return None

    current_rows = []
    current_items: Dict[str, str] = {}

    def flush_block():
        nonlocal current_rows, current_items
        if len(current_rows) >= min_rows and current_items:
            blocks.append({'items': dict(current_items), 'raw_rows': list(current_rows)})
        current_rows = []
        current_items = {}

    for line in lines:
        kv = parse_kv(line)
        if kv:
            k, v = kv
            if k not in current_items:
                current_items[k] = v
            current_rows.append(line)
        else:
            # 断开
            flush_block()
    flush_block()

    return blocks

def extract_text_from_docx(docx_path: str) -> List[str]:
    """从Word文件中按段落提取文本"""
    doc = docx.Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text_to_segments(text: str) -> List[str]:
    """将长文本分割为段落，支持多种检测模式"""
    # 检查是否启用整页检测
    if hasattr(default_config, 'PAGE_LEVEL_DETECTION') and default_config.PAGE_LEVEL_DETECTION:
        # 整页检测模式：将整页作为一个段落
        return [text.strip()] if text.strip() else []
    
    # 检查检测模式
    detection_mode = getattr(default_config, 'DETECTION_MODE', 'paragraph')
    
    if detection_mode == 'page':
        # 整页模式：不分割，直接返回
        return [text.strip()] if text.strip() else []
    elif detection_mode == 'sentence':
        # 句子模式：按句子分割
        return _split_by_sentences(text)
    else:
        # 段落模式（默认）：智能段落分割
        return _split_by_paragraphs(text)

def _split_by_sentences(text: str) -> List[str]:
    """按句子分割文本"""
    import re
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 按句号、问号、感叹号等分割句子
        line_sentences = re.split(r'[。！？；]', line)
        for sent in line_sentences:
            sent = sent.strip()
            if sent and len(sent) >= 50:  # 过滤过短的句子
                sentences.append(sent)
    return sentences

def _split_by_paragraphs(text: str) -> List[str]:
    """按段落分割文本，保持语义完整性"""
    segments = []
    current_segment = ""
    min_length = default_config.MIN_SEGMENT_LENGTH
    max_length = default_config.MAX_SEGMENT_LENGTH

    # 按句子分割，保持语义完整性
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # 按句号、问号、感叹号等分割句子
        import re
        line_sentences = re.split(r'[。！？；]', line)
        for sent in line_sentences:
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    for sentence in sentences:
        # 超长句子直接添加
        if len(sentence) > max_length:
            if current_segment:
                segments.append(current_segment.strip())
                current_segment = ""
            segments.append(sentence)
            continue

        # 合并到当前段落
        current_segment = current_segment + sentence if current_segment else sentence

        # 达到最小长度时添加到结果，但优先在句号处分割
        if len(current_segment) >= min_length:
            # 如果当前句子以句号结尾，立即分割
            if sentence.endswith(('。', '！', '？', '；')):
                segments.append(current_segment.strip())
                current_segment = ""
            # 如果接近最大长度，也进行分割
            elif len(current_segment) >= max_length * 0.8:
                segments.append(current_segment.strip())
                current_segment = ""

    # 添加最后一个段落
    if current_segment:
        segments.append(current_segment.strip())

    return segments

def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """去除停用词，保留中文文本完整性"""
    return ''.join(char for char in text if char not in stopwords)

def detect_grammar_errors(text: str) -> List[str]:
    """检测文本中的语法错误"""
    return []

def is_order_changed(text1: str, text2: str) -> bool:
    """判断两个文本是否为字符集合相等但顺序不同"""
    return set(text1) == set(text2) and text1 != text2

def is_stopword_evade(text1: str, text2: str, stopwords: List[str]) -> bool:
    """判断两个文本去除停用词后字符集合相等但原文本不同"""
    filtered1 = [char for char in text1 if char not in stopwords]
    filtered2 = [char for char in text2 if char not in stopwords]
    return set(filtered1) == set(filtered2) and text1 != text2

def remove_numbers(text: str) -> str:
    """移除文本中的所有数字"""
    return re.sub(r'[0-9]+', '', text)

def is_synonym_evade(text1: str, text2: str) -> bool:
    """判断两个文本是否通过替换同义词来规避相似度检测"""
    # 构建反向同义词映射
    reverse_synonyms = {}
    for key, syn_list in SYNONYMS.items():
        for syn in syn_list:
            if syn not in reverse_synonyms:
                reverse_synonyms[syn] = []
            reverse_synonyms[syn].append(key)
    
    # 简单分词
    def simple_tokenize(text):
        return re.findall(r'[\u4e00-\u9fa50-9a-zA-Z]+', text)
    
    tokens1 = simple_tokenize(text1)
    tokens2 = simple_tokenize(text2)
    
    # 检查词数差异
    if abs(len(tokens1) - len(tokens2)) / max(len(tokens1), len(tokens2)) > default_config.SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD:
        return False
    
    # 检查同义词替换
    synonym_count = 0
    for i in range(min(len(tokens1), len(tokens2))):
        token1, token2 = tokens1[i], tokens2[i]
        if token1 == token2:
            continue
        
        # 检查是否是同义词
        is_syn = (
            (token1 in SYNONYMS and token2 in SYNONYMS[token1]) or
            (token2 in SYNONYMS and token1 in SYNONYMS[token2]) or
            (token1 in reverse_synonyms and token2 in reverse_synonyms[token1]) or
            (token2 in reverse_synonyms and token1 in reverse_synonyms[token2])
        )
        
        if is_syn:
            synonym_count += 1
    
    # 判断是否为规避行为
    total_tokens = max(len(tokens1), len(tokens2))
    return total_tokens > 0 and synonym_count / total_tokens > 0.1



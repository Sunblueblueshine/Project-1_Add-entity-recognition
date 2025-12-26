"""
相似度分析服务模块
"""
import os
import gc
import json
import time
import uuid
import logging
import threading
import psutil
from typing import List, Dict, Any, Optional, Tuple
from threading import Thread, Lock, Timer

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from app.service.text_utils import (
    extract_text_from_pdf, extract_text_from_docx, split_text_to_segments,
    remove_stopwords, detect_grammar_errors, is_order_changed,
    is_stopword_evade, remove_numbers, is_synonym_evade,
    extract_kv_tables_from_text
)
from app.config.similarity_config import default_config
from app.config.terms_config import COMMON_TERMS
from app.service.entity_rec_service import default_entity_rec_service

# 初始化提取文本保存目录
EXTRACTED_TEXTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../extracted_texts'))
os.makedirs(EXTRACTED_TEXTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class SimilarityService:
    def __init__(self, config=None):
        """初始化相似度分析服务"""
        self.config = config or default_config
        self.storage_dir = self.config.STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 任务管理
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.running_tasks = 0
        self.task_lock = Lock()
        self.task_queue: List[Tuple[str, str, List[str]]] = []
        
        # 模型和资源初始化
        self.model = self._load_text2vec_model()
        self.stopwords = self._load_stopwords()
        self.stopwords_list = list(self.stopwords)
        
        # 缓存配置值
        self._cache_config_values()
        
        # 内存管理
        self.cleanup_counter = 0
        
    def _cache_config_values(self):
        """缓存配置值以提高性能"""
        self.max_concurrent_tasks = self.config.MAX_CONCURRENT_TASKS
        # OCR模式下不需要表格处理模式
        self.min_text_length = self.config.MIN_TEXT_LENGTH
        self.tender_similarity_threshold = self.config.TENDER_SIMILARITY_THRESHOLD
        self.bid_similarity_threshold = self.config.BID_SIMILARITY_THRESHOLD
        self.batch_size = self.config.BATCH_SIZE
        self.max_task_timeout = self.config.MAX_TASK_TIMEOUT
        self.memory_cleanup_interval = self.config.MEMORY_CLEANUP_INTERVAL
        self.similarity_top_k = self.config.SIMILARITY_TOP_K
        
        # OCR模式下不需要表格阈值
        
        # 规避检测阈值
        self.high_similarity_threshold = self.config.HIGH_SIMILARITY_THRESHOLD
        self.very_high_similarity_threshold = self.config.VERY_HIGH_SIMILARITY_THRESHOLD
        self.common_term_count_threshold = self.config.COMMON_TERM_COUNT_THRESHOLD
        self.semantic_evade_lower_threshold = self.config.SEMANTIC_EVADE_LOWER_THRESHOLD
        self.semantic_evade_upper_threshold = self.config.SEMANTIC_EVADE_UPPER_THRESHOLD
        self.cleanup_interval = self.memory_cleanup_interval
        
        
    def _load_text2vec_model(self) -> SentenceTransformer:
        """加载本地文本向量模型"""
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../local_text2vec_model'))
        model = SentenceTransformer(model_path)
        if self.config.ENABLE_GPU and torch.cuda.is_available():
            model = model.to('cuda')
        return model

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """对文本列表进行向量化"""
        texts_without_numbers = [remove_numbers(text) for text in texts]
        vectors = self.model.encode(texts_without_numbers, convert_to_numpy=True, show_progress_bar=False)
        # 确保返回float32类型的数组，FAISS需要这种类型
        return vectors.astype(np.float32)

    def _load_stopwords(self) -> set:
        """加载停用词表"""
        stopwords_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stopwords.txt'))
        if os.path.exists(stopwords_path):
            with open(stopwords_path, encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def save_file(self, file_bytes: bytes, filename: str) -> str:
        """保存上传的文件到本地存储"""
        safe_name = filename.replace("..", "_").replace("/", "_")
        save_path = os.path.join(self.storage_dir, safe_name)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        return save_path

    def start_analysis(self, tender_file_path: str, bid_file_paths: List[str]) -> str:
        """启动分析任务"""
        task_id = str(uuid.uuid4())
        task_info = {
            "status": "pending",
            "result": None,
            "progress": {"current": 0, "total": 1},
            "created_time": time.time(),
            "file_info": {
                "tender_file": os.path.basename(tender_file_path),
                "bid_files": [os.path.basename(p) for p in bid_file_paths],
                "bid_count": len(bid_file_paths)
            }
        }
        self.tasks[task_id] = task_info
        self.task_queue.append((task_id, tender_file_path, bid_file_paths))
        self._process_queue()
        return task_id
    
    def start_analysis_from_extracted_texts(self, extracted_data: Dict[str, Any]) -> str:
        """
        基于已提取的文本数据启动相似度分析任务
        """
        task_id = str(uuid.uuid4())
        
        # 验证提取数据的完整性
        if not self._validate_extracted_data(extracted_data):
            raise ValueError("提取数据格式不正确或缺少必要字段")
        
        # 创建任务
        self.tasks[task_id] = {
            "status": "pending",
            "progress": {"current": 0, "total": 0},
            "result": None,
            "error": None,
            "created_at": time.time(),
            "extracted_data": extracted_data
        }
        
        # 启动分析任务
        thread = Thread(target=self._analyze_extracted_task, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"基于提取文本启动相似度分析任务: {task_id}")
        return task_id
    
    def _validate_extracted_data(self, extracted_data: Dict[str, Any]) -> bool:
        """验证提取数据的完整性"""
        required_fields = ['tender_texts', 'bid_files']
        
        # 检查必要字段
        for field in required_fields:
            if field not in extracted_data:
                logger.error(f"提取数据缺少必要字段: {field}")
                return False
        
        # 检查招标文件文本
        if not extracted_data.get('tender_texts') or len(extracted_data['tender_texts']) == 0:
            logger.error("招标文件文本为空")
            return False
        
        # 检查投标文件
        bid_files = extracted_data.get('bid_files', [])
        if not bid_files or len(bid_files) == 0:
            logger.error("投标文件为空")
            return False
        
        # 检查每个投标文件的文本
        for bid_file in bid_files:
            if not bid_file.get('texts') or len(bid_file['texts']) == 0:
                logger.warning(f"投标文件 {bid_file.get('file_name', 'unknown')} 文本为空")
        
        return True
    
    def _analyze_extracted_task(self, task_id: str) -> None:
        """分析已提取文本的相似度任务"""
        try:
            logger.info(f"开始分析已提取文本任务: {task_id}")
            start_time = time.time()
            
            # 获取提取数据
            extracted_data = self.tasks[task_id]["extracted_data"]
            
            # 处理招标文件文本 - 按页面格式处理
            tender_texts = extracted_data['tender_texts']
            tender_segments = []
            for text_data in tender_texts:
                # 直接使用页面文本，不进行额外的切分
                page_text = text_data.get('text', '')
                if page_text and len(page_text.strip()) >= self.min_text_length:
                    clean_text = remove_stopwords(page_text, list(self.stopwords))
                    if clean_text:
                        grammar_errors = detect_grammar_errors(clean_text)
                        
                        #新增，实体识别
                        entities = []
                        if hasattr(self,'entity_service'):
                            entities = self.entity_service.extract_entities(page_text)
                        else:
                            #如果没有实体服务，则调用默认服务
                            entities = default_entity_rec_service.extract_entities(page_text)
                            """try:
                                entities = default_entity_rec_service.recognize(clean_text)
                            except Exception as e:
                                logger.warning(f"实体识别失败（招标页 {text_data.get('page', 1)}）: {e}")
                            """
                        
                        tender_segments.append({
                            'page': text_data.get('page', 1),
                            'text': clean_text,
                            'grammar_errors': grammar_errors,
                            'is_table_cell': False,
                            'entities': entities,#新增，实体识别结果
                            'original_text': page_text #新增，保存原始文本
                        })

                        #新增：将实体识别结果也保存回原始数据，用于最终输出
                        if 'entities' not in text_data:
                            text_data['entities'] = entities
            
            # 处理投标文件文本
            bid_files_data = extracted_data['bid_files']
            bid_file_paths = [bid_file['file_name'] for bid_file in bid_files_data]
            filtered_bid_segments = []
            filtered_bid_vecs = []
            
            for bid_file_data in bid_files_data:
                file_segments = []
                for text_data in bid_file_data['texts']:
                    # 直接使用页面文本，不进行额外的切分
                    page_text = text_data.get('text', '')
                    if page_text and len(page_text.strip()) >= self.min_text_length:
                        clean_text = remove_stopwords(page_text, list(self.stopwords))
                        if clean_text:
                            grammar_errors = detect_grammar_errors(clean_text)
                            
                            #新增：实体识别
                            entities = []
                            if hasattr(self,'entity_service'):
                                entities = self.entity_service.extract_entities(page_text)
                            else:
                                entities = default_entity_rec_service.extract_entities(page_text)
                            file_segments.append({
                                'page': text_data.get('page', 1),
                                'text': clean_text,
                                'grammar_errors': grammar_errors,
                                'is_table_cell': False,
                                'entities': entities,#新增，实体识别结果
                                'original_text': page_text #新增，保存原始文本
                            })
                            #新增：将实体识别结果也保存回原始数据，用于最终输出
                            if 'entities' not in text_data:
                                text_data['entities'] = entities
                if file_segments:
                    # 向量化
                    texts = [seg['text'] for seg in file_segments]
                    vecs = self._batch_encode(texts)
                    filtered_bid_segments.append(file_segments)
                    filtered_bid_vecs.append(vecs)
            
            # 执行相似度比较
            details = self._compare_bid_documents(
                filtered_bid_segments, filtered_bid_vecs, bid_file_paths, task_id
            )
            
            # 收集语法错误
            common_grammar_errors = self._collect_common_grammar_errors(
                filtered_bid_segments, bid_file_paths
            )
            
            # 生成结果
            # 修改，在生成结果时包含实体识别
            self._generate_result(
                task_id, details, common_grammar_errors, filtered_bid_segments,
                bid_file_paths, start_time #extracted_data #传入提取数据
            )
            
        except Exception as e:
            logger.error(f"分析已提取文本任务失败: {str(e)}", exc_info=True)
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["error"] = str(e)

    def _process_queue(self) -> None:
        """处理任务队列，控制并发数"""
        with self.task_lock:
            while self.running_tasks < self.max_concurrent_tasks and self.task_queue:
                task_id, tender_file_path, bid_file_paths = self.task_queue.pop(0)
                self.running_tasks += 1
                thread = Thread(target=self._analyze_task, args=(task_id, tender_file_path, bid_file_paths))
                thread.daemon = True
                thread.start()

    def _extract_and_segment(self, file_path: str) -> List[Dict[str, Any]]:
        """
        文本提取与分段：支持 PDF、Word 文档。
        提取表格单元格并进行单独处理。
        上下文感知逻辑基于同一页内的文本内容，不跨页面进行上下文合并。
        返回：[{page, text, grammar_errors, is_table_cell, row, col, table_idx}]
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            pages = extract_text_from_pdf(file_path)
            segments = []
            SEG_DELIM = ('。', '！', '？', '.', '!', '?')
            auto_merge = getattr(self.config, 'AUTO_MERGE_CROSS_PAGE_PARAGRAPH', False)

            last_unfinished = None
            for idx, page_data in enumerate(pages):
                page_num = page_data['page_num']
                segs = []
                for seg in split_text_to_segments(page_data['text']):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg and len(clean_seg) >= self.min_text_length:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segs.append({
                            'page': page_num,
                            'text': clean_seg,
                            'grammar_errors': grammar_errors,
                            'is_table_cell': False
                        })

                # 表格（KV块）提取为独立段落，便于专门比较
                if getattr(self.config, 'ENABLE_TABLE_DETECTION', False):
                    kv_blocks = extract_kv_tables_from_text(page_data.get('text', '') or '')
                    for blk in kv_blocks:
                        # 将items拼接为统一规范文本
                        items = blk.get('items', {})
                        if not items:
                            continue
                        kv_text = '; '.join([f"{k}={v}" for k, v in items.items()])
                        if kv_text and len(kv_text) >= self.min_text_length // 3:
                            grammar_errors = []
                            segs.append({
                                'page': page_num,
                                'text': kv_text,
                                'grammar_errors': grammar_errors,
                                'is_table_cell': True,
                                'is_table_segment': True,
                                'table_items': items
                            })

                # 自动合并跨页段落
                if auto_merge:
                    if last_unfinished and segs:
                        # 前一页未完+本页首段
                        segs[0]['text'] = last_unfinished['text'] + segs[0]['text']
                        segs[0]['grammar_errors'] = list(set(last_unfinished['grammar_errors'] + segs[0]['grammar_errors']))
                        last_unfinished = None
                    # 针对当页最后一段判断是否结尾
                    if segs:
                        last_seg = segs[-1]
                        # 如最后无常见句末标点，则记作未完成
                        if not last_seg['text'].rstrip().endswith(SEG_DELIM):
                            last_unfinished = segs.pop()
                segments.extend(segs)
            # 处理最后遗留
            if auto_merge and last_unfinished:
                segments.append(last_unfinished)
            return segments
        elif ext in ['.doc', '.docx']:
            paras = extract_text_from_docx(file_path)
            segments = []
            for i, para in enumerate(paras):
                for seg in split_text_to_segments(para):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg and len(clean_seg) >= self.min_text_length:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segments.append({
                            'page': i+1,
                            'text': clean_seg,
                            'grammar_errors': grammar_errors,
                            'is_table_cell': False
                        })
            return segments
        else:
            return []

    def _create_faiss_index(self, dim: int) -> faiss.Index:
        """
        创建FAISS索引，根据配置和硬件情况决定是否使用GPU。
        """
        if self.config.ENABLE_GPU and hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                return faiss.GpuIndexFlatIP(res, dim)
            except Exception:
                # 如果GPU初始化失败，回退到CPU
                return faiss.IndexFlatIP(dim)
        else:
            return faiss.IndexFlatIP(dim)

    def _faiss_max_sim(self, query_vecs: np.ndarray, base_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用faiss进行最大相似度检索"""
        if len(base_vecs) == 0 or len(query_vecs) == 0:
            return np.array([]), np.array([])
        
        # 确保向量数据类型为float32
        base_vecs = base_vecs.astype(np.float32)
        query_vecs = query_vecs.astype(np.float32)
        
        dim = base_vecs.shape[1]
        index = self._create_faiss_index(dim)
        faiss.normalize_L2(base_vecs)
        faiss.normalize_L2(query_vecs)
        index.add(base_vecs)
        sims, idxs = index.search(query_vecs, self.similarity_top_k)
        return sims.flatten(), idxs.flatten()

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """分批向量化文本，防止内存溢出"""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        
        # 获取模型的实际输出维度
        first_batch = texts[:1]
        sample_vec = self._encode_texts(first_batch)
        dim = sample_vec.shape[1] if len(sample_vec.shape) > 1 else 384
        
        # 预分配结果数组
        total_len = len(texts)
        result = np.zeros((total_len, dim), dtype=np.float32)
        
        # 处理第一个批次
        result[0] = sample_vec[0] if sample_vec.shape[0] > 0 else np.zeros(dim, dtype=np.float32)
        
        # 处理剩余批次
        for i in range(1, total_len, self.batch_size):
            end_idx = min(i + self.batch_size, total_len)
            batch = texts[i:end_idx]
            vecs = self._encode_texts(batch)
            # 确保数据类型为float32
            result[i:end_idx] = vecs.astype(np.float32)
            
            # 定期清理内存
            if (i // self.batch_size) % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
        
        return result

    def _cleanup_memory(self) -> None:
        """
        清理内存和GPU缓存。
        优化点：使用计数器减少gc调用频率，提高性能
        """
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_interval:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup_counter = 0

    def _save_extracted_texts(self, task_id: str, tender_file_path: str, tender_segments: List[Dict[str, Any]],
                              bid_file_paths: List[str], bid_segments_list: List[List[Dict[str, Any]]]) -> None:
        """
        保存任务中提取的所有文本到extracted_texts目录
        """
        try:
            # 准备要保存的数据
            extracted_data = {
                "task_id": task_id,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tender_file": os.path.basename(tender_file_path),
                "tender_texts": [],
                "bid_files": []
            }
            # 添加招标文件提取的文本（含实体识别）
            for seg in tender_segments:
                page_obj = {
                     "page": seg["page"],
                    "text": seg["text"],
                    "is_table_cell": seg["is_table_cell"]
                }
                try:
                    page_obj["entities"] = default_entity_rec_service.recognize(seg["text"])
                except Exception as e:
                    logger.warning(f"实体识别失败（招标页 {seg.get('page')}）: {e}")
                    page_obj["entities"] = []
                extracted_data["tender_texts"].append(page_obj)
            
            # 添加每个投标文件提取的文本
            for i, bid_path in enumerate(bid_file_paths):
                bid_data = {
                    "file_name": os.path.basename(bid_path),
                    "texts": []
                }
                
                if i < len(bid_segments_list):
                    for seg in bid_segments_list[i]:
                        page_obj = {
                            "page": seg["page"],
                            "text": seg["text"],
                            "is_table_cell": seg["is_table_cell"]
                        }
                        try:
                            page_obj["entities"] = default_entity_rec_service.recognize(seg["text"])
                        except Exception as e:
                            logger.warning(f"实体识别失败（投标文件 {bid_data['file_name']} 页 {seg.get('page')}）: {e}")
                            page_obj["entities"] = []
                        bid_data["texts"].append(page_obj)
                
                extracted_data["bid_files"].append(bid_data)
            
            # 生成文件名：包含任务ID和时间戳
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            tender_filename = os.path.splitext(os.path.basename(tender_file_path))[0]
            safe_filename = tender_filename.replace('..', '_').replace('/', '_').replace('\\', '_')
            filename = f"{safe_filename}_{task_id}_{timestamp}.json"
            file_path = os.path.join(EXTRACTED_TEXTS_DIR, filename)
            
            # 保存为JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"提取的文本已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存提取的文本失败: {str(e)}")
            
    def _log_resource(self, tag: str, extra: Optional[dict] = None) -> None:
        """
        记录当前进程的内存和 CPU 占用，便于监控。
        """
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # MB
        cpu = process.cpu_percent(interval=0.1)
        log_msg = f"[{tag}] mem={mem:.1f}MB cpu={cpu:.1f}%"
        if extra:
            log_msg += f" | {extra}"
        logger.info(log_msg)

    def _analyze_task(self, task_id: str, tender_file_path: str, bid_file_paths: List[str]) -> None:
        """
        任务主流程：
        1. 招标文件分段并向量化
        2. 投标文件分段并向量化
        3. 剔除与招标文件高度相似的投标片段
        4. 投标文件间两两比对，检测相似片段及规避行为
        5. 统计语法错误，找出多份文件中相同错误
        """
        start_time = time.time()
        self._log_resource('TASK_START', {'task_id': task_id})
        timeout_flag = {'timeout': False}
        
        # 设置超时计时器（仅当MAX_TASK_TIMEOUT > 0时启用）
        timer = None
        if self.max_task_timeout > 0:
            def timeout_callback() -> None:
                timeout_flag['timeout'] = True
                self.tasks[task_id]["status"] = "timeout"
                self.tasks[task_id]["result"] = {"error": f"任务执行超时 ({self.max_task_timeout}秒)"}
                self._log_resource('TASK_TIMEOUT', {'task_id': task_id})
                
            timer = threading.Timer(self.max_task_timeout, timeout_callback)
            timer.daemon = True
            timer.start()
            logger.info(f"任务 {task_id} 启用超时保护: {self.max_task_timeout}秒")
        else:
            logger.info(f"任务 {task_id} 无超时限制，支持大规模文档分析")
        
        try:
            # 初始化进度跟踪
            self.tasks[task_id]["status"] = "running"
            total_comparisons = len(bid_file_paths) * (len(bid_file_paths) - 1) // 2
            self.tasks[task_id]["progress"] = {"current": 0, "total": total_comparisons}

            # 1. 处理招标文件
            tender_segments, tender_vecs = self._process_tender_document(tender_file_path)
            
            # 2. 处理投标文件
            bid_segments_list, bid_vecs_list = self._process_bid_documents(bid_file_paths)
            
            # 保存提取的文本
            self._save_extracted_texts(task_id, tender_file_path, tender_segments, bid_file_paths, bid_segments_list)
            
            # 3. 剔除与招标文件高度相似的投标片段
            filtered_bid_segments, filtered_bid_vecs = self._filter_bid_segments(
                bid_segments_list, bid_vecs_list, tender_vecs
            )
            
            # 4. 投标文件间两两比对，检测相似片段及规避行为
            details = self._compare_bid_documents(
                filtered_bid_segments, filtered_bid_vecs, bid_file_paths, task_id
            )
            
            # 5. 统计所有投标文件分段的语法错误
            common_grammar_errors = self._collect_common_grammar_errors(
                filtered_bid_segments, bid_file_paths
            )
            
            # 6. 生成分析结果
            self._generate_result(
                task_id, details, common_grammar_errors, filtered_bid_segments, 
                bid_file_paths, start_time
            )
            
            if self.max_task_timeout > 0 and timeout_flag['timeout']:
                return
        except Exception as e:
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["result"] = {"error": str(e), "details": str(e.__traceback__)}
            self._log_resource('TASK_ERROR', {'task_id': task_id, 'error': str(e)})
            # 记录异常堆栈
            logging.error(f"Task {task_id} failed with error: {str(e)}", exc_info=True)
        finally:
            try:
                if timer is not None:
                    timer.cancel()
            except Exception:
                pass
            # 最终清理内存
            self._cleanup_memory()
            with self.task_lock:
                self.running_tasks -= 1
                self._process_queue()
            self._log_resource('TASK_FINALLY', {'task_id': task_id})

    def _process_tender_document(self, tender_file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        处理招标文件：提取文本并向量化
        """
        tender_segments = self._extract_and_segment(tender_file_path)
        tender_texts = [seg['text'] for seg in tender_segments]
        # 确保段落和向量数量一致
        if tender_texts:
            tender_vecs = self._batch_encode(tender_texts)
            if isinstance(tender_vecs, list):
                tender_vecs = np.array(tender_vecs)
        else:
            # 如果没有文本，创建空的向量数组，维度与段落数量一致
            tender_vecs = np.zeros((0, 384))
        return tender_segments, tender_vecs
        
    def _process_bid_documents(self, bid_file_paths: List[str]) -> Tuple[List[List[Dict[str, Any]]], List[np.ndarray]]:
        """
        处理投标文件：提取文本并向量化
        """
        bid_segments_list = []
        bid_vecs_list = []
        for bid_path in bid_file_paths:
            segs = self._extract_and_segment(bid_path)
            texts = [s['text'] for s in segs]
            # 确保段落和向量数量一致
            if texts:
                vecs = self._batch_encode(texts)
                if isinstance(vecs, list):
                    vecs = np.array(vecs)
            else:
                # 如果没有文本，创建空的向量数组，维度与段落数量一致
                vecs = np.zeros((0, 384))
            
            bid_segments_list.append(segs)
            bid_vecs_list.append(vecs)
            del segs, texts, vecs
            self._cleanup_memory()
        return bid_segments_list, bid_vecs_list
        
    def _filter_bid_segments(self, 
                            bid_segments_list: List[List[Dict[str, Any]]],
                            bid_vecs_list: List[np.ndarray],
                            tender_vecs: np.ndarray) -> Tuple[List[List[Dict[str, Any]]], List[np.ndarray]]:
        """
        剔除投标文件中与招标文件高度相似的片段
        """
        filtered_bid_segments = []
        filtered_bid_vecs = []
        
        for segs, vecs in zip(bid_segments_list, bid_vecs_list):
            if isinstance(vecs, list):
                vecs = np.array(vecs)
            keep_idx = []
            
            # 如果招标文件向量为空，则保留所有投标片段
            if tender_vecs.shape[0] == 0:
                keep_idx = list(range(len(segs)))
            else:
                for i in range(0, len(segs), self.batch_size):
                    batch_vecs = vecs[i:i+self.batch_size]
                    if batch_vecs.shape[0] > 0:
                        sims, _ = self._faiss_max_sim(batch_vecs, tender_vecs)
                        # 只保留与招标文件相似度低于阈值的片段
                        max_seg_idx = len(segs) - 1
                        keep_idx.extend([min(i+ii, max_seg_idx) for ii, sim in enumerate(sims) if sim < self.tender_similarity_threshold])
                    # 定期清理内存
                    if (i // self.batch_size) % self.memory_cleanup_interval == 0:
                        self._cleanup_memory()
            
            # 处理空keep_idx的情况，并确保索引有效
            if keep_idx:
                # 确保所有索引都在有效范围内
                valid_keep_idx = [idx for idx in keep_idx if 0 <= idx < len(segs)]
                if valid_keep_idx:
                    filtered_bid_segments.append([segs[idx] for idx in valid_keep_idx])
                    filtered_bid_vecs.append(vecs[valid_keep_idx])
                else:
                    # 如果没有有效索引，添加空的段落和对应的空向量
                    filtered_bid_segments.append([])
                    filtered_bid_vecs.append(np.zeros((0, vecs.shape[1] if vecs.shape[0] > 0 else 384)))
            else:
                # 如果没有保留的索引，添加空的段落和对应的空向量
                filtered_bid_segments.append([])
                filtered_bid_vecs.append(np.zeros((0, vecs.shape[1] if vecs.shape[0] > 0 else 384)))
            del segs, vecs
            self._cleanup_memory()
        
        return filtered_bid_segments, filtered_bid_vecs
        
    def _compare_bid_documents(self, 
                              filtered_bid_segments: List[List[Dict[str, Any]]],
                              filtered_bid_vecs: List[np.ndarray],
                              bid_file_paths: List[str],
                              task_id: str) -> List[Dict[str, Any]]:
        """
        投标文件间两两比对，检测相似片段及规避行为
        """
        details = []
        progress_cnt = 0
        total_comparisons = len(filtered_bid_segments) * (len(filtered_bid_segments) - 1) // 2
        self.tasks[task_id]["progress"] = {"current": 0, "total": total_comparisons}
        
        for i in range(len(filtered_bid_segments)):
            for j in range(i+1, len(filtered_bid_segments)):
                segs_i, vecs_i = filtered_bid_segments[i], filtered_bid_vecs[i]
                segs_j, vecs_j = filtered_bid_segments[j], filtered_bid_vecs[j]
                
                if isinstance(vecs_i, list):
                    vecs_i = np.array(vecs_i)
                if isinstance(vecs_j, list):
                    vecs_j = np.array(vecs_j)
                
                if vecs_i.shape[0] == 0 or vecs_j.shape[0] == 0:
                    progress_cnt += 1
                    self.tasks[task_id]["progress"]["current"] = progress_cnt
                    continue
                
                # 确保向量和段落的数量一致
                if len(segs_i) != vecs_i.shape[0] or len(segs_j) != vecs_j.shape[0]:
                    logger.warning(f"向量和段落数量不匹配: segs_i={len(segs_i)}, vecs_i={vecs_i.shape[0]}, segs_j={len(segs_j)}, vecs_j={vecs_j.shape[0]}")
                    logger.debug(f"文件i: {bid_file_paths[i] if i < len(bid_file_paths) else 'unknown'}")
                    logger.debug(f"文件j: {bid_file_paths[j] if j < len(bid_file_paths) else 'unknown'}")
                    progress_cnt += 1
                    self.tasks[task_id]["progress"]["current"] = progress_cnt
                    continue
                
                # 构建索引一次，多次查询
                dim = vecs_j.shape[1]
                index = self._create_faiss_index(dim)
                
                # 确保向量数据类型为float32
                vecs_j = vecs_j.astype(np.float32)
                faiss.normalize_L2(vecs_j)
                index.add(vecs_j)
                
                for k in range(0, vecs_i.shape[0], self.batch_size):
                    end_idx = min(k + self.batch_size, vecs_i.shape[0])
                    batch_vecs = vecs_i[k:end_idx].astype(np.float32)
                    faiss.normalize_L2(batch_vecs)
                    sims, idxs = index.search(batch_vecs, self.similarity_top_k)
                    
                    for idx_b in range(len(batch_vecs)):
                        for top_k in range(self.similarity_top_k):
                            sim = sims[idx_b][top_k]
                            max_idx = idxs[idx_b][top_k]
                            idx_i = k + idx_b
                            
                            # 确保索引有效
                            if max_idx >= len(segs_j) or idx_i >= len(segs_i):
                                logger.debug(f"索引越界: idx_i={idx_i}, len(segs_i)={len(segs_i)}, max_idx={max_idx}, len(segs_j)={len(segs_j)}")
                                continue
                            
                            text1 = segs_i[idx_i]['text']
                            text2 = segs_j[max_idx]['text']
                            
                            # OCR模式下所有文本都是普通文本，不区分表格
                            is_table_cell_i = False
                            is_table_cell_j = False
                            is_table_row_i = False
                            is_table_row_j = False
                            is_table_cell_only_i = False
                            is_table_cell_only_j = False
                              
                            # 判断是否为有效相似片段
                            is_valid = self._is_valid_similarity(
                                sim, text1, text2, 
                                is_table_cell_i, is_table_cell_j,
                                is_table_row_i, is_table_row_j,
                                is_table_cell_only_i, is_table_cell_only_j,
                                segs_i[idx_i], segs_j[max_idx]
                            )
                            
                            if is_valid:
                                # 检测规避行为
                                evade_results = self._detect_evasion_behavior(text1, text2, sim)
                                
                                # 构建相似片段详情
                                detail = self._build_similarity_detail(
                                    bid_file_paths[i], bid_file_paths[j],
                                    segs_i[idx_i], segs_j[max_idx],
                                    sim, evade_results,
                                    is_table_cell_i, is_table_cell_j,
                                    top_k
                                )
                                
                                details.append(detail)
                        # 定期清理内存
                        if (k // self.batch_size) % self.memory_cleanup_interval == 0:
                            self._cleanup_memory()
                    
                    progress_cnt += 1
                    self.tasks[task_id]["progress"]["current"] = progress_cnt
                    
                    # 清理索引资源
                    try:
                        del index
                    except:
                        pass
        
        return details
        
    def _is_valid_similarity(self, sim: float, text1: str, text2: str,
                           is_table_cell_i: bool, is_table_cell_j: bool,
                           is_table_row_i: bool, is_table_row_j: bool,
                           is_table_cell_only_i: bool, is_table_cell_only_j: bool,
                           seg_i: Dict[str, Any], seg_j: Dict[str, Any]) -> bool:
        """
        判断两个文本片段是否为有效相似
        """
        # 表格元素匹配条件
        row_match = False
        col_match = False
        table_match = False
        if (is_table_row_i and is_table_row_j) or (is_table_cell_only_i and is_table_cell_only_j):
            row_i = seg_i.get('row')
            row_j = seg_j.get('row')
            table_i = seg_i.get('table_idx')
            table_j = seg_j.get('table_idx')
            row_match = row_i == row_j
            table_match = table_i == table_j
            if is_table_cell_only_i and is_table_cell_only_j:
                col_i = seg_i.get('col')
                col_j = seg_j.get('col')
                col_match = col_i == col_j
          
        # OCR模式下统一使用普通文本阈值
        if self._is_near_identical_page(text1, text2, sim):
            current_threshold = getattr(self.config, 'NEAR_IDENTICAL_THRESHOLD', 0.95)
        else:
            current_threshold = self.bid_similarity_threshold
          
        # 判断是否为有效相似片段（OCR模式下简化判断）
        is_valid = (sim > current_threshold and 
                    len(text1) >= self.min_text_length and 
                    len(text2) >= self.min_text_length)
        
        # 额外过滤：排除相似度极高但实际是常见模板文本的情况
        if is_valid and sim > self.config.HIGH_SIMILARITY_THRESHOLD:
            # 检查是否包含大量相同的专业术语或固定表述
            term_count = sum(1 for term in COMMON_TERMS if term in text1 and term in text2)
            if term_count > self.config.COMMON_TERM_COUNT_THRESHOLD:
                # 降低极高相似度的通用模板文本的阈值要求
                is_valid = sim > self.config.VERY_HIGH_SIMILARITY_THRESHOLD
        
        return is_valid
    
    def _is_near_identical_page(self, text1: str, text2: str, similarity: float) -> bool:
        """
        判断是否为几乎相同的页面（只有盖章等微小差异）
        """
        # 基本条件：相似度很高
        if similarity < 0.9:
            return False
        
        # 长度相近
        len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        if len_diff > 0.1:  # 长度差异超过10%
            return False
        
        # 检查是否只有微小差异（如盖章、日期、签名等）
        return self._has_only_minor_differences(text1, text2)
    
    def _has_only_minor_differences(self, text1: str, text2: str) -> bool:
        """
        检查两个文本是否只有微小差异（如盖章、日期、签名等）
        """
        import re
        
        # 移除常见的微小差异
        def normalize_text(text):
            # 移除日期
            text = re.sub(r'\d{4}年\d{1,2}月\d{1,2}日', '', text)
            text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', '', text)
            # 移除时间
            text = re.sub(r'\d{1,2}:\d{2}', '', text)
            # 移除盖章标记
            text = re.sub(r'（.*?章.*?）', '', text)
            text = re.sub(r'（.*?印.*?）', '', text)
            # 移除签名
            text = re.sub(r'签名：.*', '', text)
            text = re.sub(r'签字：.*', '', text)
            # 移除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)
        
        # 计算标准化后的相似度
        if len(norm1) == 0 or len(norm2) == 0:
            return False
        
        # 简单的字符级相似度计算
        common_chars = sum(1 for c1, c2 in zip(norm1, norm2) if c1 == c2)
        max_len = max(len(norm1), len(norm2))
        char_similarity = common_chars / max_len if max_len > 0 else 0
        
        # 如果标准化后相似度很高，认为是几乎相同
        return char_similarity > 0.95
        
    def _detect_evasion_behavior(self, text1: str, text2: str, sim: float) -> Dict[str, bool]:
        """
        检测各种规避行为
        """
        # 检测语序变更规避
        order_changed = is_order_changed(text1, text2)
        
        # 检测停用词插入规避
        stopword_evade = is_stopword_evade(text1, text2, list(self.stopwords))
        
        # 检测同义词替换规避
        synonym_evade = False
        if not stopword_evade and sim < 0.95:
            synonym_evade = is_synonym_evade(text1, text2)
        
        # 检测语义保持但词汇变化较大的情况
        semantic_evade = False
        if sim > self.config.SEMANTIC_EVADE_LOWER_THRESHOLD and \
           sim < self.config.SEMANTIC_EVADE_UPPER_THRESHOLD and \
           not order_changed and not stopword_evade and not synonym_evade:
            # 可能存在更高级的规避行为
            semantic_evade = True
        
        return {
            'order_changed': order_changed,
            'stopword_evade': stopword_evade,
            'synonym_evade': synonym_evade,
            'semantic_evade': semantic_evade
        }
        
    def _build_similarity_detail(self, bid_file_i: str, bid_file_j: str,
                               seg_i: Dict[str, Any], seg_j: Dict[str, Any],
                               sim: float, evade_results: Dict[str, bool],
                               is_table_cell_i: bool, is_table_cell_j: bool,
                               top_k: int) -> Dict[str, Any]:
        """
        构建相似片段详情字典
        """
        # 检查是否为几乎相同的页面
        is_near_identical = self._is_near_identical_page(seg_i['text'], seg_j['text'], sim)
        
        detail = {
            'bid_file': os.path.basename(bid_file_i),
            'page': seg_i['page'],
            'text': seg_i['text'],
            'grammar_errors': seg_i.get('grammar_errors', []),
            'similar_with': os.path.basename(bid_file_j),
            'similar_page': seg_j['page'],
            'similarity': float(f'{sim:.4f}'),
            'is_near_identical': is_near_identical,  # 标记是否为几乎相同页面
            'similar_text': seg_j['text'],
            'similar_grammar_errors': seg_j.get('grammar_errors', []),
            'order_changed': evade_results['order_changed'],
            'stopword_evade': evade_results['stopword_evade'],
            'synonym_evade': evade_results['synonym_evade'],
            'semantic_evade': evade_results['semantic_evade'],
            'rank': top_k + 1
        }

        # 如果两侧都是表格段，附加表格相似度辅助信息
        if seg_i.get('is_table_segment') and seg_j.get('is_table_segment'):
            items_i = seg_i.get('table_items', {})
            items_j = seg_j.get('table_items', {})
            keys_i = set(items_i.keys())
            keys_j = set(items_j.keys())
            union = keys_i | keys_j
            inter = keys_i & keys_j
            param_match_rate = (len(inter) / len(union)) if union else 0.0

            # 数值一致性：相同key下，数值或文本近似
            def normalize_number(s: str) -> Optional[float]:
                try:
                    import re
                    t = s.replace(',', '')
                    m = re.search(r"-?\d+(?:\.\d+)?", t)
                    if not m:
                        return None
                    return float(m.group(0))
                except:
                    return None

            tol = getattr(self.config, 'TABLE_VALUE_TOLERANCE', 0.02)
            value_hits = 0
            value_total = len(inter)
            for k in inter:
                vi = items_i.get(k, '')
                vj = items_j.get(k, '')
                ni = normalize_number(vi)
                nj = normalize_number(vj)
                if ni is not None and nj is not None and nj != 0:
                    if abs(ni - nj) / max(abs(nj), 1e-9) <= tol:
                        value_hits += 1
                else:
                    # 退化到文本相等判断
                    if vi.strip() == vj.strip():
                        value_hits += 1
            value_match_rate = (value_hits / value_total) if value_total else 0.0

            w_text = getattr(self.config, 'TABLE_TEXT_WEIGHT', 0.4)
            w_val = getattr(self.config, 'TABLE_VALUE_WEIGHT', 0.6)
            table_similarity = float(f'{(w_text * float(detail["similarity"]) + w_val * value_match_rate):.4f}')

            detail['is_table_segment'] = True
            detail['param_match_rate'] = float(f'{param_match_rate:.4f}')
            detail['value_match_rate'] = float(f'{value_match_rate:.4f}')
            detail['table_similarity'] = table_similarity
        
        return detail
        
    def _collect_common_grammar_errors(self, 
                                     filtered_bid_segments: List[List[Dict[str, Any]]],
                                     bid_file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        收集多份文件中相同的语法错误
        """
        grammar_error_map = {}
        for file_idx, segs in enumerate(filtered_bid_segments):
            for seg in segs:
                for err in seg.get('grammar_errors', []):
                    key = (err, seg['text'])
                    if key not in grammar_error_map:
                        grammar_error_map[key] = []
                    grammar_error_map[key].append({
                        'bid_file': os.path.basename(bid_file_paths[file_idx]),
                        'page': seg['page'],
                        'text': seg['text']
                    })
        
        # 只保留出现在多份文件的相同语法错误
        common_grammar_errors = [
            {'error': k[0], 'text': k[1], 'locations': v}
            for k, v in grammar_error_map.items() if len(v) > 1
        ]
        
        return common_grammar_errors

    def _generate_result(self, task_id: str, details: List[Dict[str, Any]],
                       common_grammar_errors: List[Dict[str, Any]],
                       filtered_bid_segments: List[List[Dict[str, Any]]],
                       bid_file_paths: List[str], start_time: float) -> None:
        """
        生成分析结果并更新任务状态
        """
        # 统计相似度数量（OCR模式下不区分表格）
        total_similarity_count = len(details)
        
        summary = f'分析完成，发现{total_similarity_count}处高相似度片段，{len(common_grammar_errors)}组相同语法错误。'
        
        # 添加更多结果指标
        total_bid_files = len(bid_file_paths)
        total_segments_processed = sum(len(segs) for segs in filtered_bid_segments)
        avg_similarity_score = float(f'{np.mean([d["similarity"] for d in details]):.4f}') if details else 0.0
        max_similarity_score = float(f'{max([d["similarity"] for d in details]):.4f}') if details else 0.0
        

        #修改，获取实体统计信息
        entity_statistics = {}
        try:
            #从任务中获取提取数据
            extracted_data = self.tasks.get(task_id, {}).get('extracted_data', {})
            if extracted_data:
                entity_statistics = self._collect_entity_statistics(extracted_data)
                #将实体统计添加到summary中
                total_entities = entity_statistics.get('total_entities', 0)
                if total_entities > 0:
                    entity_types = entity_statistics.get('entity_types', {})
                    entity_type_str = "、".join(entity_types[:3])#只显示前3中，这里有保留，是否全显示
                    if len(entity_types) > 3:
                        entity_type_str += f"等{len(entity_types)}种"
                    summary += f' 共识别出{total_entities}个实体，主要包括{entity_type_str}。'
        except Exception as e:
            logger.error(f"生成实体统计信息失败: {str(e)}")
        #修改，结束

        self.tasks[task_id]["status"] = "done"
        #self.tasks[task_id]["result"] = 
        result_data = {
            "summary": summary,
            "details": details,
            "grammar_errors": common_grammar_errors,
            "total_similarity_count": total_similarity_count,
            "total_bid_files": total_bid_files,
            "total_segments_processed": total_segments_processed,
            "avg_similarity_score": avg_similarity_score,
            "max_similarity_score": max_similarity_score,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),  
            "processing_time": time.time() - start_time,      
            "task_id": task_id
        }
        

        #修改:添加实体相关信息
        if entity_statistics:
            result_data["entity_statistics"] = entity_statistics
        
        # 修改:，如果存在提取数据，将其包含在结果中（包含实体识别结果）
        extracted_data = self.tasks[task_id].get("extracted_data")
        if extracted_data:
            # 创建一个简化版的提取数据，只包含必要信息
            simplified_extracted_data = {
                "tender_texts": [],
                "bid_files": []
            }
        
            # 处理招标文件文本（包含实体）
            tender_texts = extracted_data.get("tender_texts", [])
            for text_item in tender_texts:
                simplified_item = {
                    "page": text_item.get("page", 1),
                    "text": text_item.get("text", ""),
                    "is_table_cell": text_item.get("is_table_cell", False),
                    "entities": text_item.get("entities", [])  # ✅ 包含实体识别结果
                }
                simplified_extracted_data["tender_texts"].append(simplified_item)
        
            # 处理投标文件文本（包含实体）
            bid_files = extracted_data.get("bid_files", [])
            for bid_file in bid_files:
                simplified_bid_file = {
                    "file_name": bid_file.get("file_name", ""),
                    "texts": []
                }
            
                texts = bid_file.get("texts", [])
                for text_item in texts:
                    simplified_text_item = {
                        "page": text_item.get("page", 1),
                        "text": text_item.get("text", ""),
                        "is_table_cell": text_item.get("is_table_cell", False),
                        "entities": text_item.get("entities", [])  # ✅ 包含实体识别结果
                    }
                    simplified_bid_file["texts"].append(simplified_text_item)
                simplified_extracted_data["bid_files"].append(simplified_bid_file)
            result_data["extracted_texts_with_entities"] = simplified_extracted_data
        self.tasks[task_id]["result"] = result_data
        
        elapsed = time.time() - start_time
        self._log_resource('TASK_DONE', {'task_id': task_id, 'elapsed': f'{elapsed:.1f}s'})

        # 结果持久化：保存到 saved_results 目录，命名风格与 extracted_texts 一致
        try:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_results'))
            os.makedirs(results_dir, exist_ok=True)

            # 从任务信息中拿到招标文件名（在start_analysis处保存过）
            tender_file = self.tasks.get(task_id, {}).get('file_info', {}).get('tender_file', 'result')
            tender_filename = os.path.splitext(os.path.basename(tender_file))[0]
            safe_filename = tender_filename.replace('..', '_').replace('/', '_').replace('\\', '_')
            
            #修改，在文件名中标识是否包含实体
            extracted_data = self.tasks.get(task_id).get("extracted_data")
            has_entities =False
            if extracted_data:
                #检查是否有实体数据
                for text_item in extracted_data.get("tender_texts", []):
                    if text_item.get("entities"):
                        has_entities = True
                        break       
                if not has_entities:
                    for bid_file in extracted_data.get("bid_files", []):
                        for text_item in bid_file.get("texts", []):
                            if text_item.get("entities"):
                                has_entities = True
                                break
                        if has_entities:
                            break
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')

            #修改，根据是否包含实体调整文件名
            if has_entities:
                filename = f"{safe_filename}_with_entities_{task_id}_{timestamp}.json"
            else:
                filename = f"{safe_filename}_{task_id}_{timestamp}.json"
            out_path = os.path.join(results_dir, filename)

            
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(self.tasks[task_id]['result'], f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存: {out_path}")
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        查询指定任务的结果。
        """
        if task_id not in self.tasks:
            return {"status": "not_found", "result": None}
        return self.tasks[task_id]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务的简要信息列表。
        """
        return [
            {
                "task_id": task_id,
                "status": info["status"],
                "created_time": info["created_time"],
                "file_info": info["file_info"],
                "progress": info.get("progress", {"current": 0, "total": 1})
            }
            for task_id, info in self.tasks.items()
        ]

    def cancel_task(self, task_id: str) -> bool:
        """
        取消队列中的待处理任务。
        """
        if task_id in self.tasks and self.tasks[task_id]["status"] == "pending":
            self.tasks[task_id]["status"] = "cancelled"
            # 从队列中移除
            self.task_queue = [(tid, tender, bids) for tid, tender, bids in self.task_queue if tid != task_id]
            return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> None:
        """
        清理超过指定时长的历史任务。
        """
        current_time = time.time()
        expired_tasks = [
            task_id for task_id, info in self.tasks.items()
            if current_time - info["created_time"] > max_age_hours * 3600
        ]
        for task_id in expired_tasks:
            del self.tasks[task_id]

    #新增方法
    def _collect_entity_statistics(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            all_entities = []
            entity_type_counts = {}
            
            # 统计招标文件实体
            tender_texts = extracted_data.get("tender_texts", [])
            for text_item in tender_texts:
                entities = text_item.get("entities", [])
                all_entities.extend(entities)
                for entity in entities:
                    entity_type = entity.get("entity", "未知")
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            # 统计投标文件实体
            bid_files = extracted_data.get("bid_files", [])
            for bid_file in bid_files:
                texts = bid_file.get("texts", [])
                for text_item in texts:
                    entities = text_item.get("entities", [])
                    all_entities.extend(entities)
                    for entity in entities:
                        entity_type = entity.get("entity", "未知")
                        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            # 按数量排序
            sorted_entity_counts = dict(sorted(
                entity_type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # 收集一些实体示例（每种类型最多3个）
            entity_examples = {}
            for entity_type in sorted_entity_counts.keys():
                examples = []
                # 从招标文件收集示例
                for text_item in tender_texts:
                    entities = text_item.get("entities", [])
                    for entity in entities:
                        if entity.get("entity") == entity_type:
                            examples.append(entity.get("text_content", ""))
                            if len(examples) >= 3:
                                break
                    if len(examples) >= 3:
                        break
                
                # 如果招标文件示例不足，从投标文件补充
                if len(examples) < 3:
                    for bid_file in bid_files:
                        texts = bid_file.get("texts", [])
                        for text_item in texts:
                            entities = text_item.get("entities", [])
                            for entity in entities:
                                if entity.get("entity") == entity_type:
                                    if entity.get("text_content", "") not in examples:
                                        examples.append(entity.get("text_content", ""))
                                    if len(examples) >= 3:
                                        break
                            if len(examples) >= 3:
                                break
                        if len(examples) >= 3:
                            break
                
                entity_examples[entity_type] = examples[:3]  # 最多3个示例
            
            
            return {
                "total_entities": len(all_entities),
                "entity_counts": sorted_entity_counts,
                "entity_types": list(sorted_entity_counts.keys()),  
                "entity_examples": entity_examples
            } 
            
        except Exception as e:
            logger.error(f"收集实体统计信息失败: {str(e)}")
            
            return {
                "total_entities": 0,
                "entity_counts": {},  
                "entity_types": [],   
                "entity_examples": {}
            } 
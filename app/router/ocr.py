"""
OCR文本提取API路由
"""
import os
import logging
import time
import tempfile
import requests
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from app.service.paddle_ocr_service import ocr_service
from app.service.text_utils import extract_text_from_pdf
from app.service.entity_rec_service import default_entity_rec_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR文本提取"])

# 支持的文件类型
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg', 
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}

@router.post("/upload-and-extract")
async def upload_and_extract_text(
    file: UploadFile = File(...),
    page_range: Optional[str] = Form(None, description="页面范围，如'1-3'或'1,3,5'，默认所有页面"),
    extract_tables: bool = Form(False, description="是否提取表格"),
    confidence_threshold: float = Form(0.5, description="OCR置信度阈值")
):
    """上传文件并提取文本"""
    try:
        # 检查文件类型
        if file.content_type not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file.content_type}。支持的类型: {list(SUPPORTED_FILE_TYPES.keys())}"
            )
        
        # 检查OCR服务可用性
        if not ocr_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="OCR服务不可用，请检查PaddleOCR配置"
            )
        
        # 保存上传的文件
        file_extension = SUPPORTED_FILE_TYPES[file.content_type]
        temp_file_path = f"temp_upload_{file.filename}{file_extension}"
        
        try:
            # 保存文件内容
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"文件上传成功: {file.filename}, 大小: {len(content)} 字节")
            
            # 根据文件类型处理
            if file.content_type == 'application/pdf':
                result = await process_pdf_file(temp_file_path, page_range, extract_tables, confidence_threshold)
            else:
                result = await process_image_file(temp_file_path, confidence_threshold)
            
            # 添加文件元数据
            result.update({
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file.content_type,
                "processing_time": result.get("processing_time", 0)
            })
            
            return JSONResponse(content=result)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {str(e)}")
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

async def process_pdf_file(file_path: str, page_range: Optional[str], extract_tables: bool, confidence_threshold: float) -> Dict[str, Any]:
    """处理PDF文件"""
    start_time = time.time()
    
    try:
        # 解析页面范围
        pages_to_process = parse_page_range(page_range)
        
        # 使用text_utils提取文本，直接传递页面范围
        pages_data = extract_text_from_pdf(file_path, pages_to_process)
        
        # 处理结果
        result = {
            "total_pages": len(pages_data),
            "pages": [],
            "full_text": "",
            "tables": [],
            "processing_time": 0
        }
        
        all_texts = []
        all_tables = []
        
        for page_data in pages_data:
            page_result = {
                "page_num": page_data['page_num'],
                "text": page_data.get('text', ''),
                "text_length": len(page_data.get('text', '')),
                "tables_count": len(page_data.get('tables', []))
            }
            
            if extract_tables:
                page_result["tables"] = page_data.get('tables', [])
                all_tables.extend(page_data.get('tables', []))
            
            result["pages"].append(page_result)
            all_texts.append(page_data.get('text', ''))
        
        # 合并所有文本
        result["full_text"] = '\n\n'.join(all_texts)
        result["full_text_length"] = len(result["full_text"])
        
        if extract_tables:
            result["tables"] = all_tables
            result["total_tables"] = len(all_tables)
        
        result["processing_time"] = time.time() - start_time
        return result
        
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")

async def process_image_file(file_path: str, confidence_threshold: float) -> Dict[str, Any]:
    """处理图像文件"""
    start_time = time.time()
    
    try:
        # 使用PaddleOCR处理图像
        ocr_result = ocr_service.recognize_image(file_path)
        
        result = {
            "text": ocr_result.get('text', ''),
            "text_length": len(ocr_result.get('text', '')),
            "tables": ocr_result.get('tables', []),
            "tables_count": len(ocr_result.get('tables', [])),
            "processing_time": time.time() - start_time
        }
        
        return result
        
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像处理失败: {str(e)}")

def parse_page_range(page_range: Optional[str]) -> Optional[List[int]]:
    """解析页面范围字符串"""
    if not page_range:
        return None
    
    try:
        pages = []
        for part in page_range.split(','):
            part = part.strip()
            if '-' in part:
                # 处理范围，如"1-3"
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                # 处理单页
                pages.append(int(part))
        return sorted(set(pages))  # 去重并排序
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的页面范围格式")

@router.get("/status")
async def get_ocr_status():
    """获取OCR服务状态"""
    return {
        "ocr_available": ocr_service.is_available(),
        "supported_file_types": list(SUPPORTED_FILE_TYPES.keys()),
        "max_file_size": "50MB",
        "supported_languages": ["中文", "英文"]
    }

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "ocr_service": "available" if ocr_service.is_available() else "unavailable",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

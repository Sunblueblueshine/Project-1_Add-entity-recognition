"""
相似度分析API路由
"""
import io
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import openpyxl
from app.service.similarity_service import SimilarityService
from app.service.entity_rec_service import EntityRecognitionService #新增，导入实体识别服务

router = APIRouter(prefix="/api/similarity", tags=["相似度分析"])
similarity_service = SimilarityService()
entity_service = EntityRecognitionService() #新增，实例化实体识别服务

@router.post("/upload")
async def upload_files(
    tender_file: UploadFile = File(..., description="招标文件"),
    bid_files: List[UploadFile] = File(..., description="投标文件（可多份）")
):
    """上传招标文件和投标文件"""
    try:
        tender_path = similarity_service.save_file(
            await tender_file.read(), 
            tender_file.filename or "tender_file"
        )
        bid_paths = [
            similarity_service.save_file(await f.read(), f.filename or "bid_file")
            for f in bid_files
        ]
        return {
            "msg": "文件上传成功",
            "tender_file_path": tender_path,
            "bid_file_paths": bid_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@router.post("/analyze")
async def analyze_files(
    tender_file_path: str = Form(...),
    bid_file_paths: List[str] = Form(...)
):
    """启动相似度分析任务"""
    try:
        task_id = similarity_service.start_analysis(tender_file_path, bid_file_paths)
        return {"msg": "分析任务已启动", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析任务启动失败: {str(e)}")

@router.get("/result")
async def get_result(task_id: str):
    """查询分析结果"""
    try:
        result = similarity_service.get_result(task_id)
        return {"msg": "查询成功", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/export_excel")
def export_excel(task_id: str):
    """导出分析结果为Excel文件"""
    result = similarity_service.get_result(task_id)
    if not result or result.get('status') != 'done' or not result.get('result'):
        raise HTTPException(status_code=404, detail="分析结果未找到或未完成")
    
    data = result['result']
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "雷同片段"
    
    # 添加表头
    ws.append(["投标文件", "页码", "雷同文件", "雷同页码", "相似度", "雷同文本", "语法错误", "规避行为", "几乎相同页面"])

    # 添加相似片段数据
    for d in data.get('details', []):
        evade = []
        if d.get('order_changed'): evade.append('语序规避')
        if d.get('stopword_evade'): evade.append('无意义词插入规避')
        if d.get('synonym_evade'): evade.append('同义词替换规避')
        if d.get('semantic_evade'): evade.append('语义规避')
        
        # 标记几乎相同页面
        is_near_identical = "是" if d.get('is_near_identical', False) else "否"
        

        ws.append([
            d.get('bid_file', ''), d.get('page', ''), d.get('similar_with', ''), d.get('similar_page', ''),
            d.get('similarity', ''), d.get('text', ''), '\n'.join(d.get('grammar_errors', [])), ','.join(evade), is_near_identical
        ])
    
    # 添加语法错误工作表
    ws2 = wb.create_sheet("相同语法错误")
    ws2.append(["语法错误", "片段内容", "出现位置"])
    for g in data.get('grammar_errors', []):
        locs = '\n'.join([f"{l['bid_file']} 第{l['page']}页" for l in g.get('locations', [])])
        ws2.append([g.get('error', ''), g.get('text', ''), locs])
    
    # 生成Excel文件
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf, 
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename="similarity_result_{task_id}.xlsx"'}
    )

@router.get("/tasks")
async def get_all_tasks():
    """获取所有任务列表"""
    try:
        tasks = similarity_service.get_all_tasks()
        return {"msg": "查询成功", "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.post("/cancel_task")
async def cancel_task(task_id: str = Form(...)):
    """取消指定任务"""
    try:
        success = similarity_service.cancel_task(task_id)
        if success:
            return {"msg": "任务已取消"}
        else:
            raise HTTPException(status_code=400, detail="任务无法取消或不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消失败: {str(e)}")

@router.post("/cleanup_tasks")
async def cleanup_tasks(max_age_hours: int = Form(24)):
    """清理过期任务"""
    try:
        similarity_service.cleanup_old_tasks(max_age_hours)
        return {"msg": f"已清理{max_age_hours}小时前的任务"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

@router.post("/analyze_extracted")
async def analyze_extracted_texts(extracted_data: dict):
    """
    基于已提取的文本数据进行相似度分析
    
    请求体格式:
    {
        "tender_texts": [
            {
                "page": 1,
                "text": "招标文件文本内容",
                "is_table_cell": false
            }
        ],
        "bid_files": [
            {
                "file_name": "投标文件1.pdf",
                "texts": [
                    {
                        "page": 1,
                        "text": "投标文件文本内容",
                        "is_table_cell": false
                    }
                ]
            }
        ]
    }
    """
    try:
        # 验证请求数据
        if not extracted_data:
            raise HTTPException(status_code=400, detail="请求数据不能为空")
        
        #新增开始
        # 为招标文件文本添加实体
        tender_texts = extracted_data.get('tender_texts', [])
        tender_texts_with_entities = []
        for text_item in tender_texts:
            entities = entity_service.extract_entities(text_item.get('text', ''))
            new_item = text_item.copy()
            new_item['entities'] = entities
            tender_texts_with_entities.append(new_item)
        
        # 为投标文件文本添加实体
        bid_files = extracted_data.get('bid_files', [])
        bid_files_with_entities = []
        for bid_file in bid_files:
            texts = bid_file.get('texts', [])
            texts_with_entities = []
            for text_item in texts:
                entities = entity_service.extract_entities(text_item.get('text', ''))
                new_item = text_item.copy()
                new_item['entities'] = entities
                texts_with_entities.append(new_item)
            
            new_bid_file = bid_file.copy()
            new_bid_file['texts'] = texts_with_entities
            bid_files_with_entities.append(new_bid_file)
        
        # 使用增强后的数据进行相似度分析
        enhanced_data = {
            'tender_texts': tender_texts_with_entities,
            'bid_files': bid_files_with_entities
        }
        #新增，结束

        # 启动分析任务
        task_id = similarity_service.start_analysis_from_extracted_texts(extracted_data)
        
        return {
            "msg": "基于提取文本的分析任务已启动",
            "task_id": task_id,
            "note": "文本已包含实体信息"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析任务启动失败: {str(e)}")

@router.get("/extracted_texts")
async def get_extracted_texts_list():
    """获取已提取的文本文件列表"""
    try:
        import os
        import json
        from datetime import datetime
        
        extracted_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../extracted_texts'))
        
        if not os.path.exists(extracted_dir):
            return {"msg": "查询成功", "files": []}
        
        files = []
        for filename in os.listdir(extracted_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(extracted_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 获取文件信息
                    file_info = {
                        "filename": filename,
                        "task_id": data.get('task_id', ''),
                        "timestamp": data.get('timestamp', ''),
                        "tender_file": data.get('tender_file', ''),
                        "tender_pages": len(data.get('tender_texts', [])),
                        "bid_files_count": len(data.get('bid_files', [])),
                        "file_size": os.path.getsize(file_path)
                    }
                    files.append(file_info)
                except Exception as e:
                    logger.warning(f"读取文件 {filename} 失败: {str(e)}")
        
        # 按时间排序
        files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {"msg": "查询成功", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/extracted_texts/{filename}")
async def get_extracted_texts(filename: str):
    """获取指定提取文本文件的内容"""
    try:
        import os
        import json
        
        extracted_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../extracted_texts'))
        file_path = os.path.join(extracted_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {"msg": "查询成功", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")


@router.post("/filter_result")
async def filter_result(
    result_file: UploadFile = File(..., description="分析结果JSON文件"),
    min_similarity: float = Form(0.0)
):
    """上传分析结果JSON并按相似度阈值筛选，返回筛选后的JSON。

    仅保留 similarity >= min_similarity 的记录。
    """
    try:
        import json
        import logging
        logger = logging.getLogger(__name__)
        
        raw = await result_file.read()
        data = json.loads(raw.decode("utf-8"))

        # 兼容多种JSON格式：{status, result} 或 {msg, result} 或直接结果对象
        result_obj = data
        if isinstance(data, dict):
            if "result" in data and isinstance(data.get("result"), dict):
                result_obj = data["result"]
            elif "status" in data and "result" in data.get("result", {}):
                # 嵌套结构：{status: "done", result: {result: {...}}}
                result_obj = data.get("result", {}).get("result", data)

        filtered = dict(result_obj)  # 浅拷贝
        details = list(result_obj.get("details", []))
        
        logger.info(f"接收到的JSON: details数量={len(details)}, 原始summary={result_obj.get('summary', 'N/A')}")

        new_details = [
            d for d in details
            if float(d.get('similarity', 0.0)) >= float(min_similarity)
        ]

        filtered["details"] = new_details
        
        # 重新计算统计信息
        total_similarity_count = len(new_details)
        grammar_errors_count = len(filtered.get("grammar_errors", []))
        new_summary = f'分析完成，发现{total_similarity_count}处高相似度片段，{grammar_errors_count}组相同语法错误。'
        filtered["summary"] = new_summary
        
        # 重新计算平均和最大相似度
        if new_details:
            import numpy as np
            similarities = [float(d.get('similarity', 0.0)) for d in new_details]
            filtered["avg_similarity_score"] = float(f'{np.mean(similarities):.4f}')
            filtered["max_similarity_score"] = float(f'{max(similarities):.4f}')
        else:
            filtered["avg_similarity_score"] = 0.0
            filtered["max_similarity_score"] = 0.0
        
        filtered["total_similarity_count"] = total_similarity_count
        
        logger.info(f"筛选完成: 原始{len(details)}条 -> 筛选后{total_similarity_count}条, 阈值={min_similarity}")
        logger.info(f"更新后的summary: {new_summary}")
        logger.info(f"返回数据中的summary字段: {filtered.get('summary', 'NOT FOUND')}")

        return JSONResponse(content=filtered)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"筛选失败: {str(e)}")

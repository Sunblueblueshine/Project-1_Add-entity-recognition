"""
测试实体识别集成的JSON输出格式
"""
import json
import time
import os
from typing import List, Dict, Any
from app.service.entity_rec_service import default_entity_rec_service

def test_json_output():
    """测试JSON输出格式是否符合要求"""
    
    # 模拟提取的文本数据
    tender_segments = [
        {
            "page": 2,
            "text": "邺台市襄区应急救援能力提升工程项目-\n第批设备购置\n（项目编号：HBSK2024-069）\n招标文件\n招标人：邺台市襄区应急管理局（公司）\n招标代理机构：河北槟榕工程公司（公司）\n招标时间：二零二四年十月",
            "is_table_cell": False
        },
        {
            "page": 5,
            "text": "第章 招标公告\n招合市襄区应急救援能力提升工程项目-第批设备购置\n（项目编号：HBSK2024-069）\n招标文件技术标招招标文件技术标分为三个部分。\n各投标人应按本部分的要求编制投标文件。",
            "is_table_cell": False
        }
    ]
    
    bid_file_paths = ["投标文件1.pdf", "投标文件2.pdf"]
    bid_segments_list = [
        [
            {
                "page": 1,
                "text": "投标申请书\n项目名称：邺台市襄区应急救援能力提升工程项目\n投标人：北京科技有限公司\n联系人：张三\n电话：13800138000",
                "is_table_cell": False
            }
        ],
        [
            {
                "page": 1,
                "text": "投标申请书\n项目名称：邺台市襄区应急救援能力提升工程项目\n投标人：上海新技术股份有限公司\n联系人：李四\n电话：13900139000",
                "is_table_cell": False
            }
        ]
    ]
    
    # 生成JSON数据
    extracted_data = {
        "task_id": "0e3d131d-936e-4190-b5f3-ad0440e62e60",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "tender_file": "招标文件.pdf",
        "tender_texts": [],
        "bid_files": []
    }
    
    # 处理招标文件文本
    print("=" * 80)
    print("处理招标文件文本...")
    print("=" * 80)
    for seg in tender_segments:
        page_obj = {
            "page": seg["page"],
            "text": seg["text"],
            "is_table_cell": seg["is_table_cell"]
        }
        try:
            entities = default_entity_rec_service.recognize(seg["text"])
            page_obj["entities"] = entities
            print(f"\n页码 {seg['page']} 识别实体数: {len(entities)}")
            for entity in entities:
                print(f"  - {entity['entity']}: {entity['text_content']}")
        except Exception as e:
            print(f"实体识别失败（招标页 {seg.get('page')}）: {e}")
            page_obj["entities"] = []
        extracted_data["tender_texts"].append(page_obj)
    
    # 处理投标文件文本
    print("\n" + "=" * 80)
    print("处理投标文件文本...")
    print("=" * 80)
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
                    entities = default_entity_rec_service.recognize(seg["text"])
                    page_obj["entities"] = entities
                    print(f"\n文件 {bid_data['file_name']} 页码 {seg['page']} 识别实体数: {len(entities)}")
                    for entity in entities:
                        print(f"  - {entity['entity']}: {entity['text_content']}")
                except Exception as e:
                    print(f"实体识别失败（投标文件 {bid_data['file_name']} 页 {seg.get('page')}）: {e}")
                    page_obj["entities"] = []
                bid_data["texts"].append(page_obj)
        
        extracted_data["bid_files"].append(bid_data)
    
    # 输出JSON
    print("\n" + "=" * 80)
    print("最终JSON输出（格式化）:")
    print("=" * 80)
    json_str = json.dumps(extracted_data, ensure_ascii=False, indent=2)
    print(json_str)
    
    # 验证格式
    print("\n" + "=" * 80)
    print("格式验证:")
    print("=" * 80)
    print(f"✓ 顶级字段: task_id, timestamp, tender_file, tender_texts, bid_files")
    print(f"✓ tender_texts 数量: {len(extracted_data['tender_texts'])}")
    for i, text in enumerate(extracted_data['tender_texts']):
        print(f"  - Page {text['page']}: 包含 {len(text.get('entities', []))} 个实体")
    
    print(f"✓ bid_files 数量: {len(extracted_data['bid_files'])}")
    for i, bid in enumerate(extracted_data['bid_files']):
        print(f"  - {bid['file_name']}: 包含 {len(bid['texts'])} 个页面，总实体数: {sum(len(t.get('entities', [])) for t in bid['texts'])}")
    
    return extracted_data

if __name__ == "__main__":
    test_json_output()

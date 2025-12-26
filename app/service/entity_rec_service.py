import re
from typing import List, Dict, Any

class EntityRecognitionService:
    def __init__(self):
        # 专注5种核心实体类型
        self.patterns = {
            "公司": [
                # 匹配公司名后直接跟数字的情况（常见于标题）
                r'[\u4e00-\u9fa5]{2,50}(?:有限公司|有限责任公司|股份公司|股份有限公司|公司)(?=[0-9\-—])',

                # 通用公司名，允许后面有各种字符
                r'[\u4e00-\u9fa5a-zA-Z0-9]{2,50}(?:有限公司|有限责任公司|股份公司|股份有限公司|公司)\b',

                # 烟草公司特殊格式
                r'[\u4e00-\u9fa5]{2,10}省烟草公司[\u4e00-\u9fa5]{2,10}市公司\b',
                r'[\u4e00-\u9fa5]{2,10}省[\u4e00-\u9fa5]{2,10}公司\b',

                # 匹配带空格的完整公司名
                r'[\u4e00-\u9fa5]{2,20} [\u4e00-\u9fa5]{2,30}公司\b',
            ],

           "人名": [
                # 增强正则表达式，防止"方式"被误识别 
                # 原代码：r'联系方式\s*联系\s*([\u4e00-\u9fa5]{2,4})',
                # 修改后：添加边界检查，确保提取的不是"方式"
                r'联系方式\s*联系\s*([\u4e00-\u9fa5]{2,4})(?![式样])',

                # 匹配"授权 ("（直接匹配）
                r'授权\s*([\u4e00-\u9fa5]{2,4})\s*[(（]',

                # 标准姓名标签
                r'(?:姓名|姓名[:：]\s*)[:：]?\s*([\u4e00-\u9fa5]{2,4})\b',

                # 联系人（添加边界检查）
                r'(?:联系人|联系人[:：]\s*)[:：]?\s*([\u4e00-\u9fa5]{2,4})(?:\s|$|:|：|，|,)\b',

                # 负责人（添加边界检查）
                r'(?:负责人|项目负责人)[:：]\s*([\u4e00-\u9fa5]{2,4})(?:\s|$|:|：|，|,)\b',

                # 法人代表
                r'(?:法定代表人|法人代表)[:：]\s*([\u4e00-\u9fa5]{2,4})\b',

                # 专门匹配"联系方式 联系 舒湘黔"格式
                r'(?:联系方式\s*)?联系\s+([\u4e00-\u9fa5]{2,4})\s+(?:电话|手机)[:：]',

                # 专门匹配"授权 人名(被授权"格式
                r'授权\s+([\u4e00-\u9fa5]{2,4})\s*[(（]\s*被授权',

                # 匹配"授权 人名"简单格式
                r'授权\s+([\u4e00-\u9fa5]{2,4})(?=\s|$|[(（])',

                # 匹配"人名(身份证号码"格式
                r'([\u4e00-\u9fa5]{2,4})\s*[(（]\s*(?:身份证|身份证号码)\s*[:：;；]',

                # 用户名
                r'(?:用户名|用户)[:：]\s*"?([a-zA-Z0-9_]{3,20})"?\b',

                # 新增：精确匹配"联系："格式（处理"联系：侯振宇"）
                r'联系[:：]\s*([\u4e00-\u9fa5]{2,4})(?:\s|$|，|,)',

                # 人名+称谓
                r'([\u4e00-\u9fa5]{2,4})\s*(?:先生|女士|小姐|同志)\b',

                # 括号前的人名（处理"签字代表（彭江海）"）
                r'([\u4e00-\u9fa5]{2,4})\s*(?:\(|（)\s*(?:姓名|代表)[^\)）]*(?:\)|）)',

                # 匹配"签字代表（彭江海）"这种格式
                r'签字代表\s*(?:\(|（)\s*([\u4e00-\u9fa5]{2,4})\s*(?:\)|）)',
                r'代表\s*(?:\(|（)\s*([\u4e00-\u9fa5]{2,4})\s*(?:\)|）)',

                # 匹配"联系  舒湘黔"这种空格格式
                r'联系\s+([\u4e00-\u9fa5]{2,4})\b',

                # 匹配冒号前的人名
                r'([\u4e00-\u9fa5]{2,4})\s*[:：]\s*(?:电话|手机|联系方式)',

                # 匹配明确的人名格式
                r'([\u4e00-\u9fa5]{2,4})\s*(?:签字|盖章)\b',
                
                # 新增：专门处理"签字代表 刘永凤 经理"格式 
                r'签字代表\s+([\u4e00-\u9fa5]{2,4})\s+经理',
                r'代表\s+([\u4e00-\u9fa5]{2,4})\s+经理',
            ],

            "身份证": [
                # 标准18位身份证
                r'\b\d{17}[\dXx]\b',
                r'\b\d{18}\b',
                # 带标签的身份证
                r'(?:身份证|身份证号|身份证号码|ID|公民身份号码)[:：]\s*\d{17}[\dXx]\b',
                r'(?:身份证|身份证号|身份证号码|ID|公民身份号码)[:：]\s*\d{18}\b',
            ],

            "联系方式": [
                # 手机号
                r'\b1[3-9]\d{9}\b',
                r'(?:手机|电话|联系电话|手机号码|手机号|联系方式)[:：]\s1[3-9]\d{9}\b',
                # 固话
                r'\b\d{3,4}-\d{7,8}\b',
                r'(?:电话|联系电话|办公电话|固定电话|传真)[:：]\s\d{3,4}-\d{7,8}\b',
                # 邮箱
                r'\b\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)\b',
                r'(?:邮箱|电子邮件|E-mail|email)[:：]\s\w+@\w+\.\w+\b',
            ],

            "项目编号": [
                r'(?:项目编号|编号|招标编号|采购编号|文件编号)[:：]\s*[A-Za-z0-9\-_]{5,30}\b',
                r'\b[A-Z]{2,10}\d{4}-\d{2,5}\b',
                r'\b[A-Z]{2,10}-\d{4}-\d{2,5}\b',
                r'\b[A-Z]{2,10}\d{6,10}\b',
            ]
        }

        # 公司识别黑名单
        self.company_blacklist = {
            "本公司", "为公司", "使公司", "向公司", "给公司", "对公司",
            "熟悉公司", "建立公司", "代表公司", "为本公司", "介绍公司",
            "反馈公司", "递交给公司", "同时公司", "依照公司", "增强公司",
            "客户公司", "项目公司", "服务公司", "业务公司",
            "中心", "站", "局", "院", "所", "办", "处", "部", "委", "厅",
            "楼", "商", "街", "路", "巷", "号", "栋", "单元", "室",
            "指挥中心", "供应站", "办公室", "会议室", "服务站", "工作站",
            "控股", "其总公司"
        }

        # 人名黑名单（扩展）
        self.name_blacklist = {
            "附件三", "附件四", "附件五", "附件六", "附件七", "附件八",
            "评标委员", "中标名称", "招标名称", "未中标", "确认通知",
            "问题澄清", "中标通知", "结果通知", "姓名", "联系人", "负责人",
            "法定代表人", "法人代表", "经办人", "代理人", "授权人",
            "联系电话", "联系制度", "员工总", "联系联系", "联系方式",
            "销售经理", "项目经理", "技术经理",
            # 新增黑名单词汇
            "单位负责", "印鉴签字", "签字盖章", "签字确认", "签字代表",
            "负责人", "单位", "印鉴", "签字", "盖章", "负责", "项目负责",
            # 新增：将"方式"和"经理"加入黑名单
            "方式", "经理", "代表", "职务"
        }

        # 常见姓氏列表
        self.common_surnames = set("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍卻璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公万俟司马上官欧阳夏侯诸葛闻人东方赫连皇甫尉迟公羊澹台公冶宗政濮阳淳于单于太叔申屠公孙仲孙轩辕令狐钟离宇文长孙慕容鲜于闾丘司徒司空亓官司寇仉督子车颛孙端木巫马公西漆雕乐正壤驷公良拓跋夹谷宰父谷梁晋楚阎法汝鄢涂钦段干百里东郭南门呼延归海羊舌微生岳帅缑亢况后有琴梁丘左丘东门西门商牟佘佴伯赏南宫墨哈谯笪年爱阳佟第五言福百家姓续")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体"""
        if not text or not isinstance(text, str):
            return []

        # 新增：创建清理换行符的文本副本 
        cleaned_text = self._clean_text_for_entity_recognition(text)
        
        # 保存原始文本用于位置映射
        original_text = text
        
        # 在清理后的文本上提取实体
        entities = self._extract_from_cleaned_text(cleaned_text, original_text)
        
        # 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x["start_pos"])

        return entities
    
    # 新增：专门处理被换行符截断的文本 
    def _clean_text_for_entity_recognition(self, text: str) -> str:
        """
        预处理文本，处理换行符导致的文本截断问题
        用于实体识别的文本清理
        """
        if not text:
            return ""
        
        # 处理常见的被换行符拆分的词语
        # 1. 先将所有换行符替换为特殊标记，以便处理
        lines = text.split('\n')
        
        # 2. 重建文本，处理被拆分的词语
        reconstructed = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 检查是否需要与前一行合并
            if i > 0 and reconstructed:
                prev_line = reconstructed[-1]
                
                # 定义常见的被拆分模式
                split_patterns = [
                    # (上一行结尾, 当前行开头, 完整词语)
                    ('联', '系', '联系'),
                    ('签', '字', '签字'),
                    ('方', '式', '方式'),
                    ('代', '表', '代表'),
                    ('负', '责', '负责'),
                    ('电', '话', '电话'),
                    ('手', '机', '手机'),
                    ('姓', '名', '姓名'),
                    ('联', '络', '联络'),
                    ('授', '权', '授权'),
                    ('经', '理', '经理'),
                    ('联', '系人', '联系人'),
                ]
                
                for end_char, start_char, full_word in split_patterns:
                    # 检查是否匹配拆分模式
                    if (prev_line.endswith(end_char) and 
                        (line.startswith(start_char) or 
                         (len(start_char) > 1 and line.startswith(start_char[0])))):
                        # 合并词语
                        if prev_line.endswith(end_char):
                            reconstructed[-1] = prev_line[:-len(end_char)] + full_word
                        else:
                            reconstructed[-1] = prev_line + full_word[len(end_char):]
                        
                        # 移除当前行已匹配的部分
                        if line.startswith(start_char):
                            line = line[len(start_char):]
                        elif len(start_char) > 1 and line.startswith(start_char[0]):
                            # 部分匹配的情况
                            line = line[1:]
                        break
            
            if line:
                reconstructed.append(line)
        
        # 3. 用空格连接所有行，形成连续的文本
        cleaned_text = ' '.join(reconstructed)
        
        # 4. 处理多余的空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    # 新增：在清理文本上提取实体并映射位置
    def _extract_from_cleaned_text(self, cleaned_text: str, original_text: str) -> List[Dict[str, Any]]:
        """在清理后的文本上提取实体，并映射回原始文本位置"""
        entities = []
        
        # 构建原始文本到清理文本的位置映射
        position_map = self._build_position_map(original_text, cleaned_text)
        
        # 在清理文本上使用原有的提取逻辑
        temp_entities = []
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, cleaned_text)
                    for match in matches:
                        # 提取匹配的文本
                        if match.groups():
                            # 提取第一个非空分组
                            for group in match.groups():
                                if group:
                                    matched_text = group
                                    start_pos = match.start(match.groups().index(group) + 1)
                                    end_pos = match.end(match.groups().index(group) + 1)
                                    break
                        else:
                            matched_text = match.group()
                            start_pos = match.start()
                            end_pos = match.end()

                        # 应用实体特定过滤
                        if self._is_valid_entity(entity_type, matched_text):
                            entity = {
                                "entity": entity_type,
                                "text_content": matched_text,
                                "start_pos": start_pos,  # 在清理文本中的位置
                                "end_pos": end_pos,      # 在清理文本中的位置
                                "_cleaned": True         # 标记这是在清理文本中提取的
                            }
                            temp_entities.append(entity)
                except re.error as e:
                    continue
        
        # 将清理文本中的位置映射回原始文本
        for entity in temp_entities:
            # 查找实体在原始文本中的位置
            original_positions = self._find_in_original_text(
                entity["text_content"], 
                original_text,
                position_map,
                entity["start_pos"],
                entity["end_pos"]
            )
            
            if original_positions:
                original_start, original_end = original_positions
                entities.append({
                    "entity": entity["entity"],
                    "text_content": entity["text_content"],
                    "start_pos": original_start,
                    "end_pos": original_end
                })
        
        return entities
    
    # 新增：构建位置映射关系 
    def _build_position_map(self, original_text: str, cleaned_text: str) -> Dict[int, int]:
        """
        构建原始文本位置到清理文本位置的映射
        返回：{原始位置: 清理位置}
        """
        position_map = {}
        
        o_idx = 0  # 原始文本索引
        c_idx = 0  # 清理文本索引
        
        while o_idx < len(original_text) and c_idx < len(cleaned_text):
            # 跳过原始文本中的换行符
            if original_text[o_idx] == '\n':
                o_idx += 1
                continue
            
            # 当字符匹配时建立映射
            if original_text[o_idx] == cleaned_text[c_idx]:
                position_map[o_idx] = c_idx
                o_idx += 1
                c_idx += 1
            else:
                # 处理合并的词语（如"联"+"系"->"联系"）
                # 这种情况下，清理文本可能跳过了原始文本的一些字符
                c_idx += 1
        
        return position_map
    
    # 新增：在原始文本中查找实体 
    def _find_in_original_text(self, entity_text: str, original_text: str, 
                               position_map: Dict[int, int],
                               cleaned_start: int, cleaned_end: int) -> tuple:
        """
        在原始文本中查找实体
        返回：(原始起始位置, 原始结束位置) 或 None
        """
        # 方法1：尝试直接搜索实体文本
        match = re.search(re.escape(entity_text), original_text)
        if match:
            return match.start(), match.end()
        
        # 方法2：使用位置映射
        # 查找最近的映射位置
        for offset in range(-5, 6):  # 在附近查找
            test_pos = cleaned_start + offset
            if test_pos in position_map.values():
                # 找到清理位置对应的原始位置
                for orig_pos, clean_pos in position_map.items():
                    if clean_pos == test_pos:
                        # 在原始文本中从该位置开始搜索
                        search_start = max(0, orig_pos - 10)
                        search_end = min(len(original_text), orig_pos + len(entity_text) + 10)
                        context = original_text[search_start:search_end]
                        
                        # 在上下文中搜索实体文本
                        match_in_context = re.search(re.escape(entity_text), context)
                        if match_in_context:
                            return (search_start + match_in_context.start(), 
                                    search_start + match_in_context.end())
        
        # 方法3：如果上述方法都失败，返回None
        return None

    def _is_valid_entity(self, entity_type: str, text: str) -> bool:
        """验证实体是否有效（增强版）"""
        text = text.strip()

        if entity_type == "公司":
            # 过滤黑名单词汇
            if any(black_word in text for black_word in self.company_blacklist):
                return False

            # 公司名通常至少3个字
            if len(text) < 3:
                return False

            # 不能全是数字
            if text.isdigit():
                return False

            # 必须包含公司关键词
            company_keywords = {"公司", "有限", "股份", "集团"}
            if not any(keyword in text for keyword in company_keywords):
                return False

        elif entity_type == "人名":
            # 过滤黑名单（扩展检查）
            if text in self.name_blacklist:
                return False
            
            # 增强过滤逻辑 
            # 额外过滤：包含"负责"、"单位"、"印鉴"、"签字"等词汇
            invalid_keywords = {"负责", "单位", "印鉴", "签字", "盖章", "项目", "经理", "代表", "方式"}
            if any(keyword in text for keyword in invalid_keywords):
                return False

            # 中文人名：2-4个汉字
            if re.fullmatch(r'[\u4e00-\u9fa5]+', text):
                if len(text) < 2 or len(text) > 4:
                    return False

                # 必须是以常见姓氏开头
                if text[0] not in self.common_surnames:
                    # 允许复姓
                    double_surnames = {"诸葛", "欧阳", "司马", "上官", "令狐", "皇甫"}
                    if text[:2] not in double_surnames:
                        return False
                        
                # 新增：检查是否以无效词结尾 
                invalid_endings = {"式", "理", "表", "责", "字", "章"}
                if text[-1] in invalid_endings:
                    return False
                    
            # 英文用户名：3-20字符
            elif re.fullmatch(r'[a-zA-Z0-9_]+', text):
                if len(text) < 3 or len(text) > 20:
                    return False
            else:
                return False

            # 不能包含数字
            if any(char.isdigit() for char in text):
                return False

        elif entity_type == "身份证":
            # 身份证必须是18位或17位+X
            if len(text) not in [17, 18]:
                return False

            # 简单校验
            if not re.match(r'^\d{17}[\dXx]$', text):
                return False

        return True

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重实体"""
        unique_entities = []
        seen = set()

        for entity in entities:
            # 基于内容和位置去重
            key = (entity["entity"], entity["text_content"], entity["start_pos"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities
    
    #  新增：处理相似度分析JSON的便捷方法 
    def extract_entities_from_json(self, json_data: Dict) -> List[Dict]:
        """
        从相似度分析的JSON结果中提取实体
        保持原始text字段不变
        """
        results = []
        
        if isinstance(json_data, dict):
            # 单个文档
            if "text" in json_data:
                entities = self.extract_entities(json_data["text"])
                result = {
                    "page": json_data.get("page"),
                    "text": json_data["text"],  # 保持原始text不变
                    "entities": entities
                }
                results.append(result)
                
        elif isinstance(json_data, list):
            # 多个文档
            for doc in json_data:
                if isinstance(doc, dict) and "text" in doc:
                    entities = self.extract_entities(doc["text"])
                    result = {
                        "page": doc.get("page"),
                        "text": doc["text"],  # 保持原始text不变
                        "entities": entities
                    }
                    results.append(result)
        
        return results

# 全局实例
default_entity_rec_service = EntityRecognitionService()
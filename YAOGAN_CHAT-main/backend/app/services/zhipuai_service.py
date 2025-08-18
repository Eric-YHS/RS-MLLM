import logging
from zai import ZhipuAiClient
from app.core.config import ZHIPUAI_API_KEY

logger = logging.getLogger(__name__)

class ZhipuAiService:
    def __init__(self, api_key=None):
        """
        初始化智谱AI服务客户端
        
        Args:
            api_key: API密钥，如果为None则使用配置文件中的密钥
        """
        self.api_key = api_key or ZHIPUAI_API_KEY
        self.client = ZhipuAiClient(api_key=self.api_key)
        
    def _clean_special_tags(self, text):
        """
        清理文本中的特殊标记
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return text
            
        # 移除智谱AI模型输出中的特殊标记
        tags_to_remove = [
            "<|begin_of_box|>", "<|end_of_box|>",
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|begin_of_list|>", "<|end_of_list|>",
            "<|begin_of_attribute|>", "<|end_of_attribute|>"
        ]
        
        cleaned_text = text
        for tag in tags_to_remove:
            cleaned_text = cleaned_text.replace(tag, "")
        
        # 使用正则表达式处理竖线标记
        import re
        # 移除开头的单竖线或双竖线
        cleaned_text = re.sub(r'^\s*\|\|\s*', '', cleaned_text)
        cleaned_text = re.sub(r'^\s*\|\s*', '', cleaned_text)
        # 移除结尾的单竖线或双竖线
        cleaned_text = re.sub(r'\s*\|\|\s*$', '', cleaned_text)
        cleaned_text = re.sub(r'\s*\|\s*$', '', cleaned_text)
        
        return cleaned_text
        
    async def analyze_image(self, image_base64=None, image_url=None, prompt=None, task_type=None, model="glm-4.5v", context_messages=None, task_id=None, redis_client=None):
        """
        调用智谱AI GLM-4.5v API分析图像
        
        Args:
            image_base64: Base64编码的图像（与image_url二选一）
            image_url: 图像URL（与image_base64二选一）
            prompt: 分析提示或问题
            task_type: 任务类型，用于构建系统提示
            model: 使用的模型，默认为"glm-4.5v"
            context_messages: 对话上下文消息列表
            task_id: 任务ID，用于检查任务是否被取消
            redis_client: Redis客户端，用于检查取消标志
            
        Returns:
            API响应结果
        """
        # 根据任务类型构建不同的提示
        if task_type == "description":
            system_prompt = "你是一个专业的遥感图像分析AI助手。请详细描述这张遥感图像中的内容，包括地形、建筑、植被等特征。"
        elif task_type == "detection":
            system_prompt = """你是一个专业的遥感图像分析AI助手。用户要求你在遥感图像中定位特定的目标物体。

请严格按照以下格式输出检测结果：
返回物体的坐标边界框，你必须使用以下JSON格式：
```json
{"label": "物体名称", "bbox": [x1, y1, x2, y2]}
```
或者如果有多个物体:
```json
[
  {"label": "物体1", "bbox": [x1, y1, x2, y2]},
  {"label": "物体2", "bbox": [x3, y3, x4, y4]}
]
```

说明:
- x1,y1是左上角坐标，x2,y2是右下角坐标
- 所有坐标值必须是0-1之间的相对位置
- 例如：{"label": "建筑", "bbox": [0.2, 0.3, 0.5, 0.6]}

重要提示：你的回复中必须包含且只包含有效的JSON格式坐标，不要省略或更改格式。"""
        elif task_type == "segmentation":
            system_prompt = "你是一个专业的遥感图像分析AI助手。请对这张遥感图像进行语义分割，识别不同类型的地表覆盖物（如建筑、道路、植被、水体等）。"
        else:
            system_prompt = "你是一个专业的遥感图像分析AI助手。请分析这张遥感图像并回答问题。针对用户的问题，请基于历史对话的上下文给出相关的答案。"
        
        # 准备消息内容
        content = []
        
        # 添加图像
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        elif image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建完整消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # 添加上下文消息
        if context_messages and len(context_messages) > 0:
            messages.extend(context_messages)
        
        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": content
        })
        
        try:
            logger.info(f"发送请求到智谱AI {model} API")
            
            # 检查是否支持stream模式，用于支持取消功能
            stream_mode = task_id is not None and redis_client is not None
            
            if stream_mode:
                # 使用流式响应，以便可以在中间检查取消
                response = None
                collected_content = ""
                collected_thinking = ""
                
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    thinking={"type": "enabled"},
                    stream=True
                )
                
                for chunk in stream:
                    # 检查是否有取消信号
                    if redis_client.exists(f"task_cancel:{task_id}"):
                        logger.info(f"任务 {task_id} 已被用户取消，终止API调用")
                        # 创建取消响应
                        from types import SimpleNamespace
                        response = SimpleNamespace()
                        response.choices = [SimpleNamespace()]
                        response.choices[0].message = SimpleNamespace()
                        response.choices[0].message.content = collected_content + "\n\n[用户已取消生成]"
                        response.choices[0].message.thinking = collected_thinking + "\n\n[用户已取消生成]"
                        break
                    
                    # 收集内容和思考过程
                    if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        collected_content += chunk.choices[0].delta.content
                    
                    if hasattr(chunk.choices[0].delta, "thinking") and chunk.choices[0].delta.thinking:
                        collected_thinking += chunk.choices[0].delta.thinking
                
                # 如果没有被取消，构建完整响应
                if response is None:
                    from types import SimpleNamespace
                    response = SimpleNamespace()
                    response.choices = [SimpleNamespace()]
                    response.choices[0].message = SimpleNamespace()
                    response.choices[0].message.content = collected_content
                    response.choices[0].message.thinking = collected_thinking
            else:
                # 非流式模式，一次性获取响应
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    thinking={"type": "enabled"}
                )
            
            logger.info("成功接收到API响应")
            
            # 获取消息对象
            message = response.choices[0].message
            
            # 如果有content属性，清理特殊标记
            if hasattr(message, "content"):
                message.content = self._clean_special_tags(message.content)
                
            return message
            
        except Exception as e:
            logger.error(f"调用智谱AI API时出错: {str(e)}")
            return {"error": f"API调用错误: {str(e)}"}

# 创建全局服务实例，方便直接导入使用
zhipuai_service = ZhipuAiService()

# 为了保持向后兼容性，提供与旧版相同的函数接口
async def analyze_image(image_base64, prompt, task_type):
    """
    调用智谱AI GLM-4.5v API分析图像（兼容旧版接口）
    """
    result = await zhipuai_service.analyze_image(image_base64=image_base64, prompt=prompt, task_type=task_type)
    
    # 将新SDK的返回结果格式化为旧版格式
    if hasattr(result, "content"):
        return result.content
    else:
        return result

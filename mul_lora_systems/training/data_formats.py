import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import jsonschema
from PIL import Image
import logging


logger = logging.getLogger(__name__)


class TaskType(Enum):
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QA = "visual_qa"
    VISUAL_GROUNDING = "visual_grounding"
    OBJECT_DETECTION = "object_detection"
    LAND_CLASSIFICATION = "land_classification"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    SCENE_REASONING = "scene_reasoning"
    MULTIMODAL_FUSION = "multimodal_fusion"


class ImageFormat(Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    TIF = "tif"
    TIFF = "tiff"


class Region(Enum):
    GLOBAL = "global"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    ASIA = "asia"
    AFRICA = "africa"
    OCEANIA = "oceania"
    SOUTH_AMERICA = "south_america"


@dataclass
class BoundingBox:
    x: float 
    y: float  
    width: float  
    height: float  
    label: Optional[str] = None 
    confidence: Optional[float] = None 
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundingBox':
        return cls(**data)


@dataclass
class ImageMetadata:

    resolution: str  
    sensor: str  
    acquisition_date: Optional[str] = None 
    location: Optional[str] = None  
    coordinates: Optional[Tuple[float, float]] = None  
    weather: Optional[str] = None 
    time_of_day: Optional[str] = None  
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageMetadata':
        return cls(**data)


@dataclass
class ConversationTurn:

    from_role: str 
    value: str  
    
    def to_dict(self) -> Dict:
        return {"from": self.from_role, "value": self.value}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationTurn':
        return cls(from_role=data["from"], value=data["value"])


@dataclass
class MMRS1MSample:
    image_id: str  
    image_path: str
    conversations: List[ConversationTurn] 
    metadata: ImageMetadata  
    task_types: List[TaskType]  
    region: Region  
    bounding_boxes: Optional[List[BoundingBox]] = None  
    
    def to_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "conversations": [conv.to_dict() for conv in self.conversations],
            "metadata": self.metadata.to_dict(),
            "task_types": [task.value for task in self.task_types],
            "region": self.region.value,
            "bounding_boxes": [bbox.to_dict() for bbox in self.bounding_boxes] if self.bounding_boxes else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MMRS1MSample':
        return cls(
            image_id=data["image_id"],
            image_path=data["image_path"],
            conversations=[ConversationTurn.from_dict(conv) for conv in data["conversations"]],
            metadata=ImageMetadata.from_dict(data["metadata"]),
            task_types=[TaskType(task) for task in data["task_types"]],
            region=Region(data["region"]),
            bounding_boxes=[BoundingBox.from_dict(bbox) for bbox in data["bounding_boxes"]] if data.get("bounding_boxes") else None
        )


@dataclass
class ChatEarthNetSample:
    sample_id: str  
    image_path: str 
    question: str 
    answer: str 
    task_type: TaskType  
    region: Region  
    metadata: ImageMetadata  
    difficulty: Optional[str] = None 
    temporal_info: Optional[Dict] = None 
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "task_type": self.task_type.value,
            "region": self.region.value,
            "metadata": self.metadata.to_dict(),
            "difficulty": self.difficulty,
            "temporal_info": self.temporal_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatEarthNetSample':
        return cls(
            sample_id=data["sample_id"],
            image_path=data["image_path"],
            question=data["question"],
            answer=data["answer"],
            task_type=TaskType(data["task_type"]),
            region=Region(data["region"]),
            metadata=ImageMetadata.from_dict(data["metadata"]),
            difficulty=data.get("difficulty"),
            temporal_info=data.get("temporal_info")
        )


class DatasetValidator:
    def __init__(self):
        self.mmrs_schema = self._create_mmrs_schema()
        self.chatearth_schema = self._create_chatearth_schema()
    
    def _create_mmrs_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["image_id", "image_path", "conversations", "metadata", "task_types", "region"],
            "properties": {
                "image_id": {"type": "string", "pattern": "^mmrs_\\d{6}$"},
                "image_path": {"type": "string"},
                "conversations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["from", "value"],
                        "properties": {
                            "from": {"type": "string", "enum": ["human", "gpt"]},
                            "value": {"type": "string", "minLength": 1}
                        }
                    },
                    "minItems": 1
                },
                "metadata": {
                    "type": "object",
                    "required": ["resolution", "sensor"],
                    "properties": {
                        "resolution": {"type": "string"},
                        "sensor": {"type": "string"},
                        "acquisition_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                        "location": {"type": "string"},
                        "coordinates": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "weather": {"type": "string"},
                        "time_of_day": {"type": "string"}
                    }
                },
                "task_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": [task.value for task in TaskType]},
                    "minItems": 1
                },
                "region": {"type": "string", "enum": [region.value for region in Region]},
                "bounding_boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["x", "y", "width", "height"],
                        "properties": {
                            "x": {"type": "number", "minimum": 0, "maximum": 1},
                            "y": {"type": "number", "minimum": 0, "maximum": 1},
                            "width": {"type": "number", "minimum": 0, "maximum": 1},
                            "height": {"type": "number", "minimum": 0, "maximum": 1},
                            "label": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            }
        }
    
    def _create_chatearth_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["sample_id", "image_path", "question", "answer", "task_type", "region", "metadata"],
            "properties": {
                "sample_id": {"type": "string", "pattern": "^chatearth_\\d{6}$"},
                "image_path": {"type": "string"},
                "question": {"type": "string", "minLength": 1},
                "answer": {"type": "string", "minLength": 1},
                "task_type": {"type": "string", "enum": [task.value for task in TaskType]},
                "region": {"type": "string", "enum": [region.value for region in Region]},
                "metadata": {
                    "type": "object",
                    "required": ["resolution", "sensor"],
                    "properties": {
                        "resolution": {"type": "string"},
                        "sensor": {"type": "string"},
                        "acquisition_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                        "location": {"type": "string"},
                        "coordinates": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "weather": {"type": "string"},
                        "time_of_day": {"type": "string"}
                    }
                },
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "temporal_info": {"type": "object"}
            }
        }
    
    def validate_mmrs_sample(self, sample_data: Dict) -> Tuple[bool, Optional[str]]:
        try:
            jsonschema.validate(sample_data, self.mmrs_schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
    
    def validate_chatearth_sample(self, sample_data: Dict) -> Tuple[bool, Optional[str]]:
        try:
            jsonschema.validate(sample_data, self.chatearth_schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
    
    def validate_dataset_file(self, file_path: str, dataset_type: str) -> Tuple[bool, List[str]]:
        if not os.path.exists(file_path):
            return False, [f"File not found: {file_path}"]
        
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return False, ["Dataset file must contain a list of samples"]
            
            for i, sample in enumerate(data[:100]): 
                if dataset_type.lower() == "mmrs":
                    is_valid, error = self.validate_mmrs_sample(sample)
                elif dataset_type.lower() == "chatearth":
                    is_valid, error = self.validate_chatearth_sample(sample)
                else:
                    return False, [f"Unknown dataset type: {dataset_type}"]
                
                if not is_valid:
                    errors.append(f"Sample {i}: {error}")
                
                if len(errors) >= 10:  
                    errors.append("... (more errors truncated)")
                    break
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {e}"]
        except Exception as e:
            return False, [f"Error validating file: {e}"]


class DatasetGenerator:
    
    def __init__(self):
        self.validator = DatasetValidator()
    
    def generate_mmrs_sample(self, image_id: str) -> MMRS1MSample:
        conversations = [
            ConversationTurn("human", "请描述这张遥感图像的内容。"),
            ConversationTurn("gpt", "这是一张城市遥感图像，显示了密集的建筑物、道路网络和一些绿地区域。图像中可以看到住宅区、商业区和交通基础设施。")
        ]
        
        metadata = ImageMetadata(
            resolution="0.5m",
            sensor="optical",
            acquisition_date="2023-06-15",
            location="北京市朝阳区",
            coordinates=(39.9042, 116.4074),
            weather="晴朗",
            time_of_day="上午"
        )
        
        return MMRS1MSample(
            image_id=image_id,
            image_path=f"images/{image_id}.jpg",
            conversations=conversations,
            metadata=metadata,
            task_types=[TaskType.IMAGE_CAPTIONING, TaskType.VISUAL_QA],
            region=Region.ASIA
        )
    
    def generate_chatearth_sample(self, sample_id: str) -> ChatEarthNetSample:
        metadata = ImageMetadata(
            resolution="10m",
            sensor="multispectral",
            acquisition_date="2023-07-20",
            location="德国巴伐利亚州",
            coordinates=(48.1351, 11.5820),
            weather="多云",
            time_of_day="下午"
        )
        
        return ChatEarthNetSample(
            sample_id=sample_id,
            image_path=f"images/{sample_id}.tif",
            question="这个区域的土地利用类型是什么？",
            answer="这个区域主要是农业用地，包括农田和牧场，周围有一些森林覆盖。",
            task_type=TaskType.LAND_CLASSIFICATION,
            region=Region.EUROPE,
            metadata=metadata,
            difficulty="medium"
        )
    
    def generate_sample_dataset(self, dataset_type: str, num_samples: int = 10) -> List[Dict]:
        samples = []
        
        for i in range(num_samples):
            if dataset_type.lower() == "mmrs":
                sample_id = f"mmrs_{i+1:06d}"
                sample = self.generate_mmrs_sample(sample_id)
            elif dataset_type.lower() == "chatearth":
                sample_id = f"chatearth_{i+1:06d}"
                sample = self.generate_chatearth_sample(sample_id)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            samples.append(sample.to_dict())
        
        return samples
    
    def save_sample_dataset(self, dataset_type: str, output_path: str, num_samples: int = 10):
        samples = self.generate_sample_dataset(dataset_type, num_samples)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {num_samples} {dataset_type} samples and saved to {output_path}")


def create_dataset_documentation():
    doc = """
# 遥感多模态数据集格式规范

## MMRS-1M数据集格式

MMRS-1M是一个大规模遥感多模态对话数据集，包含100万个样本。

### 数据格式

```json
{
    "image_id": "mmrs_000001",
    "image_path": "images/mmrs_000001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "请描述这张遥感图像的内容。"
        },
        {
            "from": "gpt",
            "value": "这是一张城市遥感图像，显示了密集的建筑物..."
        }
    ],
    "metadata": {
        "resolution": "0.5m",
        "sensor": "optical",
        "acquisition_date": "2023-06-15",
        "location": "北京市朝阳区",
        "coordinates": [39.9042, 116.4074],
        "weather": "晴朗",
        "time_of_day": "上午"
    },
    "task_types": ["image_captioning", "visual_qa"],
    "region": "asia",
    "bounding_boxes": [
        {
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
            "label": "building",
            "confidence": 0.95
        }
    ]
}
```

### 字段说明

- `image_id`: 图像唯一标识符，格式为 "mmrs_XXXXXX"
- `image_path`: 图像文件相对路径
- `conversations`: 对话列表，包含人类问题和AI回答
- `metadata`: 图像元数据信息
- `task_types`: 任务类型列表
- `region`: 地理区域
- `bounding_boxes`: 边界框信息（可选）

## ChatEarthNet数据集格式

ChatEarthNet是一个专注于地球观测的问答数据集，包含20万个样本。

### 数据格式

```json
{
    "sample_id": "chatearth_000001",
    "image_path": "images/chatearth_000001.tif",
    "question": "这个区域的土地利用类型是什么？",
    "answer": "这个区域主要是农业用地，包括农田和牧场...",
    "task_type": "land_classification",
    "region": "europe",
    "metadata": {
        "resolution": "10m",
        "sensor": "multispectral",
        "acquisition_date": "2023-07-20",
        "location": "德国巴伐利亚州",
        "coordinates": [48.1351, 11.5820],
        "weather": "多云",
        "time_of_day": "下午"
    },
    "difficulty": "medium",
    "temporal_info": {
        "season": "summer",
        "year": 2023
    }
}
```

### 字段说明

- `sample_id`: 样本唯一标识符，格式为 "chatearth_XXXXXX"
- `image_path`: 图像文件相对路径
- `question`: 问题文本
- `answer`: 答案文本
- `task_type`: 任务类型
- `region`: 地理区域
- `metadata`: 图像元数据信息
- `difficulty`: 难度级别（easy/medium/hard）
- `temporal_info`: 时序信息（可选）

## 支持的任务类型

- `image_captioning`: 图像描述
- `visual_qa`: 视觉问答
- `visual_grounding`: 视觉定位
- `object_detection`: 目标检测
- `land_classification`: 地物分类
- `temporal_analysis`: 时序分析
- `scene_reasoning`: 场景推理
- `multimodal_fusion`: 多模态融合

## 支持的地理区域

- `global`: 全球
- `europe`: 欧洲
- `north_america`: 北美
- `asia`: 亚洲
- `africa`: 非洲
- `oceania`: 大洋洲
- `south_america`: 南美洲

## 图像格式要求

- 支持格式: JPG, JPEG, PNG, TIF, TIFF
- 分辨率范围: 0.1m - 30m
- 图像大小: 建议512x512或更高
- 颜色空间: RGB或多光谱

## 数据质量要求

1. 所有文本内容必须准确、完整
2. 图像路径必须有效
3. 元数据信息必须真实可靠
4. 边界框坐标必须归一化到[0,1]范围
5. 时间信息必须符合ISO 8601格式
"""
    
    return doc


def test_data_formats():
    print("测试数据格式定义和验证...")
    print("-" * 50)
    
    # 创建生成器和验证器
    generator = DatasetGenerator()
    validator = DatasetValidator()
    
    # 生成示例样本
    mmrs_sample = generator.generate_mmrs_sample("mmrs_000001")
    chatearth_sample = generator.generate_chatearth_sample("chatearth_000001")
    
    print("生成的MMRS-1M样本:")
    print(json.dumps(mmrs_sample.to_dict(), indent=2, ensure_ascii=False))
    print("\n" + "-" * 30 + "\n")
    
    print("生成的ChatEarthNet样本:")
    print(json.dumps(chatearth_sample.to_dict(), indent=2, ensure_ascii=False))
    print("\n" + "-" * 30 + "\n")
    
    # 验证样本格式
    is_valid, error = validator.validate_mmrs_sample(mmrs_sample.to_dict())
    print(f"MMRS-1M样本验证: {'通过' if is_valid else '失败'}")
    if error:
        print(f"错误: {error}")
    
    is_valid, error = validator.validate_chatearth_sample(chatearth_sample.to_dict())
    print(f"ChatEarthNet样本验证: {'通过' if is_valid else '失败'}")
    if error:
        print(f"错误: {error}")
    
    print("\n数据格式测试完成!")


if __name__ == "__main__":
    test_data_formats()
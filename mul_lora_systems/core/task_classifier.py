import json
import os
from typing import Dict, List, Optional, Tuple
import re


class TaskClassifier:
    def __init__(self, config_path: str = "mul_lora_systems/configs/task_mapping.json"):
        self.config_path = config_path
        self.task_mappings = self._load_config()
        self.keywords = self.task_mappings.get("keywords", {})
        self.task_priority = self.task_mappings.get("task_priority", [])
        self.default_lora = self.task_mappings.get("default_lora", "mmrs_chatearth_lora")
        self.confidence_threshold = self.task_mappings.get("confidence_threshold", 0.1)
        self.max_keywords_match = self.task_mappings.get("max_keywords_match", 3)
        
        if not self.keywords:
            default_config = self._get_default_mappings()
            self.keywords = default_config["keywords"]
            self.task_priority = default_config["task_priority"]

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using default mappings")
            return self._get_default_mappings()

    def _get_default_mappings(self) -> Dict:
        return {
            "keywords": {
                "target_detection": [
                    ["count", 1.5], ["statistics", 1.5], ["how many", 1.2],
                    ["quantity", 1.2], ["detect", 1.0], ["recognize", 1.0], 
                    ["target", 0.8], ["building", 0.6], ["vehicle", 0.6]
                ],
                "image_caption": [
                    ["describe", 1.5], ["content", 1.0], ["scene", 0.8], ["view", 0.8],
                    ["what", 0.3], ["this is", 0.3], ["see", 0.5], ["show", 0.5], ["contain", 0.5]
                ],
                "spatial_grounding": [
                    ["location", 1.5], ["position", 1.5], ["coordinate", 1.5], ["where is", 1.2], 
                    ["where", 1.2], ["direction", 1.0], ["place", 1.0], ["area", 0.8], ["center", 0.6]
                ],
                "land_classification": [
                    ["classification", 1.5], ["type", 1.2], ["kind", 1.2], ["land feature", 1.0], 
                    ["land", 1.0], ["cover", 1.0], ["usage", 0.8], ["function", 0.8]
                ],
                "scene_reasoning": [
                    ["reasoning", 1.5], ["analysis", 1.2], ["judge", 1.0], ["trend", 1.2], ["change", 1.0],
                    ["why", 1.5], ["reason", 1.2], ["result", 1.0], ["impact", 1.0]
                ],
                "multimodal_fusion": [
                    ["fusion", 1.5], ["comprehensive", 1.2], ["synergy", 1.2], ["combine", 1.0],
                    ["text", 0.8], ["speech", 0.8], ["data", 0.6]
                ]
            },
            "task_priority": [
                "target_detection", "spatial_grounding", "image_caption",
                "land_classification", "scene_reasoning", "multimodal_fusion"
            ],
            "default_lora": "mmrs_chatearth_lora",
            "confidence_threshold": 0.1,
            "max_keywords_match": 3
        }

    def classify_task(self, question: str, image_path: Optional[str] = None) -> str:
        if not question or not question.strip():
            return self.default_lora
        
        question_lower = question.lower().strip()
        task_scores = self._calculate_task_scores(question_lower)
        best_task = self._select_best_task(task_scores)
        
        return best_task

    def _calculate_task_scores(self, question: str) -> Dict[str, float]:
        task_scores = {}
        jitter = (hash(question) % 1000) / 10000.0
        
        for task_type, keywords_config in self.keywords.items():
            score = 0.0
            matched_keywords_count = 0
            
            for item in keywords_config:
                if isinstance(item, list) and len(item) == 2:
                    keyword, weight = item
                else:
                    keyword, weight = item, 1.0
                
                keyword_lower = keyword.lower()
                
                if keyword_lower in question:
                    score += weight
                    matched_keywords_count += 1
                
                if matched_keywords_count >= self.max_keywords_match:
                    break
            
            if score > 0:
                task_specific_jitter = (jitter + hash(task_type) % 100 / 1000.0) % 0.1
                score += task_specific_jitter

            task_scores[task_type] = round(score, 3)
            
        return task_scores

    def _select_best_task(self, task_scores: Dict[str, float]) -> str:
        valid_tasks = {
            task: score for task, score in task_scores.items() 
            if score >= self.confidence_threshold
        }
        
        if not valid_tasks:
            if task_scores:
                best_task_by_score = max(task_scores, key=task_scores.get)
                if task_scores[best_task_by_score] > 0:
                    return best_task_by_score
            return self.default_lora
        
        priority_map = {task: i for i, task in enumerate(self.task_priority)}
        
        sorted_tasks = sorted(
            valid_tasks.items(),
            key=lambda item: (item[1], -priority_map.get(item[0], len(priority_map))),
            reverse=True
        )
        
        return sorted_tasks[0][0]

    def get_task_confidence(self, question: str) -> Dict[str, float]:
        if not question or not question.strip():
            return {}
        
        question_lower = question.lower().strip()
        task_scores = self._calculate_task_scores(question_lower)
        
        return task_scores

    def classify_with_confidence(self, question: str, image_path: Optional[str] = None) -> Tuple[str, float]:
        task_type = self.classify_task(question, image_path)
        confidence_scores = self.get_task_confidence(question)
        confidence = confidence_scores.get(task_type, 0.0)
        
        return task_type, confidence

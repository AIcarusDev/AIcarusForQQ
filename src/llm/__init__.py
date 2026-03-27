"""llm — LLM 调用链顶层包

子包结构：
  llm.core   — LLM 引擎（provider 适配、调用核心、限流、schema、JSON 修复）
  llm.prompt — Prompt 构建（xml_builder、activity_log、memory、unread_builder）
  llm.media  — 媒体资源（image_cache、sticker_collection、vision_bridge）
  llm.session — 会话管理（保留顶层，被引用最广）
"""

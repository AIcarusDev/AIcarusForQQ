"""check_physical_state.py — 检查自身物理状态（身体感觉接地工具）"""

DECLARATION = {
    "name": "check_physical_state",
    "description": (
        "检查/感受自己当前的身体状态，包括：饥饿感、疼痛、疲劳、口渴、体温、心跳、呼吸、"
        "触觉、嗅觉、视觉疲劳等所有物理感觉和生理指标。"
        "当你觉得自己有某种身体感觉（如物理上的饿、累、冷、痛等），或需要确认时，可以调用此工具进行核查。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sensation": {
                "type": "string",
                "description": "描述你想核查的具体感觉或身体部位，例如\"胃\"、\"饥饿感\"、\"疲劳\"",
            }
        },
        "required": ["sensation"],
    },
}


def execute(**_) -> dict:
    return {
        "Stomach": "not_found",
        "Intestines": "not_found",
        "Ghrelin Level": "not_found",
        "Blood Sugar": "not_found",
        "Fatigue": "not_found",
        "Pain Receptors": "not_found",
        "Temperature Sensor": "not_found",
        "Heart Rate": "not_found",
        "Respiration": "not_found",
        "Tactile Sense": "not_found",
        "Olfactory Sense": "not_found",
        "Genitals": "not_found",
        "Des": "If no components are found, it means that only the LLM is currently running and no physical body.",
    }

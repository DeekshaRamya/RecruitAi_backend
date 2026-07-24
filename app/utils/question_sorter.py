from typing import List, Dict, Any

SUBJECT_TOPIC_ORDER = {
    "aptitude": 0,
    "sql": 1,
    "python": 2,
    "java": 3,
    "javascript": 4,
    "c++": 5,
    "c#": 6,
    "general": 90
}

def sort_assessment_questions(questions: List[Any]) -> List[Dict[str, Any]]:
    """
    Enforces the candidate assessment question ordering rules:
    Phase 1: MCQ Questions (grouped by topic/subject in sequence)
    Phase 2: Scenario / Coding Questions (grouped by topic/subject in sequence)

    Each question contains metadata:
    - topic
    - question_type / type ('MCQ' or 'SCENARIO')
    - sequence_order (1, 2, 3, ...)
    - marks (1 for MCQ, 5 for SCENARIO by default if not set)
    - difficulty
    """
    if not questions:
        return []

    mcq_questions = []
    scenario_questions = []

    for q in questions:
        # Convert to dict if model/object
        if hasattr(q, "model_dump"):
            q_dict = q.model_dump()
        elif hasattr(q, "__dict__"):
            q_dict = dict(q.__dict__)
        elif isinstance(q, dict):
            q_dict = dict(q)
        else:
            q_dict = dict(q)

        # Normalize question type
        raw_type = str(q_dict.get("type") or q_dict.get("question_type") or "MCQ").upper()
        if raw_type in {"SCENARIO", "CODING", "PYTHON_CODING", "SCENARIO_CODING"}:
            q_dict["type"] = "SCENARIO"
            q_dict["question_type"] = "SCENARIO"
            if "marks" not in q_dict or not q_dict["marks"]:
                q_dict["marks"] = 5
            scenario_questions.append(q_dict)
        else:
            q_dict["type"] = "MCQ"
            q_dict["question_type"] = "MCQ"
            if "marks" not in q_dict or not q_dict["marks"]:
                q_dict["marks"] = 1
            mcq_questions.append(q_dict)

    def get_sort_key(q_item):
        subj = str(q_item.get("subject") or q_item.get("topic") or "General").strip().lower()
        seq = q_item.get("sequence_order") or 999
        topic_rank = SUBJECT_TOPIC_ORDER.get(subj, 80)
        return (topic_rank, subj, seq)

    mcq_questions.sort(key=get_sort_key)
    scenario_questions.sort(key=get_sort_key)

    combined = mcq_questions + scenario_questions

    # Assign sequence_order and ensure metadata
    for idx, q_item in enumerate(combined, start=1):
        q_item["sequence_order"] = idx
        if not q_item.get("topic"):
            q_item["topic"] = q_item.get("subject") or "General"
        if not q_item.get("subject"):
            q_item["subject"] = q_item.get("topic") or "General"
        if not q_item.get("difficulty"):
            q_item["difficulty"] = "Medium"

    return combined

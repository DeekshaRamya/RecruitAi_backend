import json
import logging
from typing import Dict, Any, List
from app.core.azure_openai import AzureOpenAIClient
from app.schemas.assessment import AssessmentGenerateRequest

logger = logging.getLogger("recruitai-backend.azure_openai_service")

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAIClient()

    async def generate_questions(self, request: AssessmentGenerateRequest) -> List[Dict[str, Any]]:
        """
        Builds the prompt, calls the Azure OpenAI API client, cleans the response,
        and parses it into a Python dictionary.
        """
        # 1. Dynamically construct prompt
        prompt = self._build_prompt(request)
        logger.info(f"Generated Prompt:\n{prompt}")

        system_message = (
            "You are an AI assessment generator. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )

        # 2. Invoke Azure OpenAI via the client
        raw_response = await self.client.generate_chat_completion(prompt, system_message)
        
        # 3. Clean and parse JSON
        cleaned_json = self._clean_json(raw_response)
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Azure OpenAI response as JSON. Raw: {raw_response}. Error: {e}")
            raise ValueError("Azure OpenAI response is not valid JSON") from e

        if "questions" not in data or not isinstance(data["questions"], list):
            logger.error(f"Azure response JSON is missing 'questions' list key. Parsed data: {data}")
            raise ValueError("Azure OpenAI response is missing the 'questions' list key")

        return data["questions"]

    def _build_prompt(self, request: AssessmentGenerateRequest) -> str:
        """
        Dynamically constructs the prompt based on subjects, topics, question count, and difficulty.
        """
        topics_details = []
        for subject in request.subjects:
            for topic in subject.topics:
                topics_details.append(
                    f"- Subject: '{subject.name}', Topic: '{topic.name}' => Generate EXACTLY: MCQ Questions = {topic.mcqCount}, Scenario Questions = {topic.scenarioCount}"
                )
        topics_str = "\n".join(topics_details)

        prompt = f"""Generate technical recruitment assessment questions based on the following configurations:

Selected Difficulty: {request.difficulty}

Subjects & Topics Configuration:
{topics_str}

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. Do NOT include any explanations, introduction, markdown headers, or footnotes. Only return the JSON object matching the requested schema.
3. Every question must have EXACTLY four options in the "options" list. Do not leave options empty or null.
4. Every question must have EXACTLY one "correctAnswer". The "correctAnswer" must match one of the four options character-for-character (exact string match). Do not leave "correctAnswer" empty or null.
5. All generated questions must have the difficulty field set to: "{request.difficulty}".
6. Do NOT generate duplicate options within a single question. All four options must be unique.
7. Do NOT generate duplicate questions. Ensure each question tests a distinct aspect of the topic.
8. For MCQ Questions (type: "MCQ"):
   - Must contain: "subject", "topic", "type": "MCQ", "difficulty", "question", "options" (exactly 4 unique options), "correctAnswer".
9. For Scenario-Based Questions (type: "SCENARIO"):
   - Must contain: "subject", "topic", "type": "SCENARIO", "difficulty", "scenario" (a detailed programming/business scenario context), "question", "options" (exactly 4 unique options), "correctAnswer".

Response Schema:
{{
  "questions": [
    {{
      "subject": "Subject Name",
      "topic": "Topic Name",
      "type": "MCQ",
      "difficulty": "{request.difficulty}",
      "question": "Question text...",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctAnswer": "Option A"
    }},
    {{
      "subject": "Subject Name",
      "topic": "Topic Name",
      "type": "SCENARIO",
      "difficulty": "{request.difficulty}",
      "scenario": "Scenario context...",
      "question": "Question text based on scenario...",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctAnswer": "Option B"
    }}
  ]
}}"""
        return prompt

    def _clean_json(self, raw_response: str) -> str:
        """
        Cleans markdown formatting if returned by the model.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

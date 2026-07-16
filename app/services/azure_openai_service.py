import json
import logging
from typing import Dict, Any, List
from app.core.azure_openai import AzureOpenAIClient
from app.schemas.assessment import AssessmentGenerateRequest
from app.schemas.interview import InterviewGenerateRequest, InterviewEvaluateRequest

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
3. For MCQ Questions (type: "MCQ"):
   - Must contain: "subject", "topic", "type": "MCQ", "difficulty", "question", "options" (exactly 4 unique options), "correctAnswer".
   - The "correctAnswer" must match one of the four options character-for-character (exact string match). Do not leave "correctAnswer" empty or null.
   - Do NOT generate duplicate options within a single question. All four options must be unique.
4. For Scenario-Based Questions (type: "SCENARIO"):
   - Must contain: "subject", "topic", "type": "SCENARIO", "difficulty", "scenario" (a detailed programming/business/logic scenario context), "question" (the actual task/problem statement for the scenario), "correctAnswer" (the complete expected solution code, query, or text), "exampleInput" (a sample input format or data string), "exampleOutput" (the expected sample output response string).
   - Must NOT have options (set "options" to null or omit the options field).
5. All generated questions must have the difficulty field set to: "{request.difficulty}".
6. Do NOT generate duplicate questions. Ensure each question tests a distinct aspect of the topic.

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
      "options": null,
      "correctAnswer": "The complete expected code or query solution",
      "exampleInput": "sample input details",
      "exampleOutput": "sample output details"
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

    async def generate_interview_scenario(self, request: InterviewGenerateRequest) -> Dict[str, Any]:
        """
        Builds the prompt for interview scenario generation, calls the Azure OpenAI API,
        cleans the response, and parses it into a Python dictionary.
        """
        prompt = self._build_interview_prompt(request)
        logger.info(f"Generated Interview Prompt:\n{prompt}")

        system_message = (
            "You are an expert technical interviewer and mentor. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )

        raw_response = await self.client.generate_chat_completion(prompt, system_message)
        cleaned_json = self._clean_json(raw_response)

        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Azure OpenAI response as JSON. Raw: {raw_response}. Error: {e}")
            raise ValueError("Azure OpenAI response is not valid JSON") from e

        # Ensure all required keys exist
        required_keys = {"category", "topic", "difficulty", "scenario", "question", "answer", "explanation", "followUps"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            logger.error(f"Azure response JSON is missing keys: {missing_keys}. Parsed data: {data}")
            raise ValueError(f"Azure OpenAI response is missing keys: {', '.join(missing_keys)}")

        return data

    def _build_interview_prompt(self, request: InterviewGenerateRequest) -> str:
        """
        Constructs the prompt for interview scenario generation.
        """
        prompt = f"""Generate a beginner-level practical coding/query interview question, answer, and explanation based on the following:

Category: {request.category}
Topic: {request.topic}
Difficulty: {request.difficulty}

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. The scenario and question must be extremely clear, practical, and beginner-friendly.
3. For SQL categories, the question MUST ask the candidate to write an SQL query to solve the scenario (e.g., "Write an SQL query to display all employee names from the Employee table."), and the "answer" MUST contain only the complete, valid SQL query.
4. For Python categories, the question MUST ask the candidate to write Python code (e.g., "Write a Python program to print numbers from 1 to 10 using a for loop."), and the "answer" MUST contain only the complete, working Python code.
5. The "explanation" must be in simple English, explaining the solution in one or two clear sentences (e.g., "This loop prints the numbers from 1 to 10.").
6. Keep the code or query in the "answer" clean and simple for beginners.
7. Include 2 or 3 follow-up questions that an interviewer might ask next to test depth in the "followUps" array.
8. The JSON structure must match the schema below exactly.

Response Schema:
{{
  "category": "{request.category}",
  "topic": "{request.topic}",
  "difficulty": "{request.difficulty}",
  "scenario": "A brief real-world context (e.g. 'You have an Employee table.' or 'You need to write a simple iteration script.')",
  "question": "The direct question/task (e.g. 'Write an SQL query to display all employee names.')",
  "answer": "The Python code or SQL query solution.",
  "explanation": "A very simple, short explanation of the solution.",
  "followUps": [
    {{
      "question": "Follow-up question 1",
      "suggestedAnswer": "Suggested answer for follow-up 1"
    }},
    {{
      "question": "Follow-up question 2",
      "suggestedAnswer": "Suggested answer for follow-up 2"
    }}
  ]
}}"""
        return prompt

    async def evaluate_interview_answer(self, request: InterviewEvaluateRequest) -> Dict[str, Any]:
        """
        Builds the prompt for interview answer evaluation, calls the Azure OpenAI API,
        cleans the response, and parses it into a Python dictionary.
        """
        prompt = self._build_evaluation_prompt(request)
        logger.info(f"Generated Evaluation Prompt:\n{prompt}")

        system_message = (
            "You are an expert technical interviewer. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )

        raw_response = await self.client.generate_chat_completion(prompt, system_message)
        cleaned_json = self._clean_json(raw_response)

        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Azure OpenAI response as JSON. Raw: {raw_response}. Error: {e}")
            raise ValueError("Azure OpenAI response is not valid JSON") from e

        # Ensure all required keys exist
        required_keys = {"isCorrect", "score", "evaluation", "improvementSuggestions"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            logger.error(f"Azure response JSON is missing keys: {missing_keys}. Parsed data: {data}")
            raise ValueError(f"Azure OpenAI response is missing keys: {', '.join(missing_keys)}")

        return data

    def _build_evaluation_prompt(self, request: InterviewEvaluateRequest) -> str:
        """
        Constructs the prompt for evaluating a candidate's answer.
        """
        prompt = f"""Evaluate the candidate's answer for the following coding scenario interview question:

Category: {request.category}
Topic: {request.topic}
Scenario: {request.scenario}
Question: {request.question}

Reference Correct Answer:
{request.correctAnswer}

Candidate's Answer:
{request.userAnswer}

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. Review the candidate's answer carefully. Point out any syntax errors, logical errors, edge case failures, or performance issues.
3. Be fair. If the candidate's answer is correct or logically equivalent to the reference answer, set "isCorrect" to true (even if the style is slightly different). If there are significant syntax or logical errors, set "isCorrect" to false.
4. Grade the answer on a scale of 0 to 100 in "score".
5. Provide a constructive, polite critique in "evaluation" in simple, clear English. Explain why it is correct or incorrect, and highlight exactly where the issues are.
6. Provide specific tips on how to improve the code/query (e.g. time/space complexity, formatting, edge cases, SQL index or join optimizations) in "improvementSuggestions".
7. The JSON structure must match the schema below exactly.

Response Schema:
{{
  "isCorrect": true,
  "score": 85,
  "evaluation": "Constructive critique of syntax, logic, and mistakes in simple English.",
  "improvementSuggestions": "Actionable suggestions on how to improve or optimize the solution."
}}"""
        return prompt


    async def evaluate_assessment_answer(self, question: str, scenario: str, correct_answer: str, candidate_answer: str) -> Dict[str, Any]:
        """
        Evaluates a candidate's descriptive or scenario-based assessment answer using Azure OpenAI.
        """
        prompt = f"""Evaluate the candidate's answer to the following scenario-based assessment question:

Scenario: {scenario}
Question: {question}
Reference Correct Answer:
{correct_answer}

Candidate's Answer:
{candidate_answer}

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. Evaluate based on correctness, completeness, technical accuracy, keywords, and logical explanation.
3. Be fair. Score from 0 to 100.
4. "status" must be exactly one of: "Correct" (for scores >= 80), "Partially Correct" (for scores between 40 and 79), or "Incorrect" (for scores < 40).
5. Provide constructive feedback, strengths, and improvements in simple English.
6. The JSON structure must match the schema below exactly.

Response Schema:
{{
  "score": 85,
  "status": "Correct",
  "feedback": "Constructive evaluation feedback.",
  "strengths": "What the candidate did well.",
  "improvements": "What could be improved."
}}"""
        system_message = (
            "You are an AI assessment evaluator. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )

        raw_response = await self.client.generate_chat_completion(prompt, system_message)
        cleaned_json = self._clean_json(raw_response)

        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Azure OpenAI evaluation response as JSON. Raw: {raw_response}. Error: {e}")
            return {
                "score": 0,
                "status": "Incorrect",
                "feedback": "AI evaluation failed to parse response.",
                "strengths": "None",
                "improvements": "None"
            }

        return {
            "score": data.get("score", 0),
            "status": data.get("status", "Incorrect"),
            "feedback": data.get("feedback", "No feedback provided."),
            "strengths": data.get("strengths", "No strengths highlighted."),
            "improvements": data.get("improvements", "No improvement areas identified.")
        }




Resume Text:
{resume_text}

CRITICAL RULES AND CONSTRAINTS:
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. The feedback points in "resume_analysis" must be a list of strings, each 1 sentence long, highlight strengths, experience, or areas of expertise.
3. If email or name is not found in the text, use "N/A" or try your best to estimate.
4. Ensure all scores are integers.
5. Do NOT include any explanations, introduction, markdown headers, or footnotes. Only return the JSON object matching the requested schema.

Response Schema:
{{
  "name": "Candidate Name",
  "email": "candidate@example.com",
  "resume_score": 85,
  "python_score": 80,
  "sql_score": 75,
  "aptitude_score": 70,
  "english_score": 85,
  "resume_analysis": [
    "Demonstrates professional experience in Python development.",
    "Showcases hands-on knowledge in database structure and SQL queries.",
    "Clear structure and structured presentation of credentials."
  ]
}}"""
        system_message = (
            "You are an expert AI resume reviewer. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )
        
        raw_response = await self.client.generate_chat_completion(prompt, system_message)
        cleaned_json = self._clean_json(raw_response)
        
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Azure OpenAI response as JSON. Raw: {raw_response}. Error: {e}")
            raise ValueError("Azure OpenAI response is not valid JSON") from e
            
        # Validate keys and types
        required_keys = {{"name", "email", "resume_score", "python_score", "sql_score", "aptitude_score", "english_score", "resume_analysis"}}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            logger.error(f"Azure response JSON is missing keys: {{missing_keys}}. Parsed data: {{data}}")
            raise ValueError(f"Azure OpenAI response is missing keys: {{', '.join(missing_keys)}}")
            
        return data

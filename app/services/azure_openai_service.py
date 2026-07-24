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
        Dynamically constructs the prompt based on selected subjects and percentage distributions.
        AI determines ideal total question count (15-30) if not specified.
        """
        num_subjects = len(request.subjects)
        target_total = request.totalQuestions or (15 if num_subjects == 1 else (20 if num_subjects == 2 else 25))

        mcq_count = round(target_total * request.questionDistribution.mcq / 100.0)
        scenario_count = target_total - mcq_count

        easy_count = round(target_total * request.difficultyDistribution.easy / 100.0)
        medium_count = round(target_total * request.difficultyDistribution.medium / 100.0)
        hard_count = target_total - (easy_count + medium_count)

        subjects_str = ", ".join(request.subjects)

        prompt = f"""Generate technical recruitment assessment questions based on the following configurations:

Selected Subjects: {subjects_str}

AUTOMATIC QUESTION COUNT SELECTION:
- You (the AI) must automatically determine the ideal total number of questions (between 15 and 30 questions depending on the selected subjects and assessment objective).
- Target approximately {target_total} total questions.

Question Type Distribution Ratios (Strictly respect these percentages across the generated question set):
- MCQ Questions (type "MCQ"): ~{mcq_count} questions ({request.questionDistribution.mcq}%)
- Scenario-Based Questions (type "SCENARIO"): ~{scenario_count} questions ({request.questionDistribution.scenario}%)

Difficulty Level Distribution Ratios (Strictly respect these percentages across the generated question set):
- Easy: ~{easy_count} questions ({request.difficultyDistribution.easy}%)
- Medium: ~{medium_count} questions ({request.difficultyDistribution.medium}%)
- Hard: ~{hard_count} questions ({request.difficultyDistribution.hard}%)

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. Do NOT include any explanations, introduction, markdown headers, or footnotes outside the JSON.
3. Distribute the questions balanced across the selected subjects: {subjects_str}. The "subject" attribute of each question MUST be one of {request.subjects}.
4. Avoid duplicate or repetitive questions. Generate realistic, interview-quality questions.
5. For MCQ Questions (type: "MCQ"):
   - Must contain: "subject", "topic", "type": "MCQ", "difficulty" ("Easy", "Medium", or "Hard"), "question", "options" (array of exactly 4 unique strings), "correctAnswer" (must match one option exactly), "explanation".
6. For Scenario-Based Questions (type: "SCENARIO"):
   - Must contain: "subject", "topic", "type": "SCENARIO", "difficulty" ("Easy", "Medium", or "Hard"), "scenario" (real-world problem statement context), "question" (the main task), "problemStatement" (detailed problem description), "candidateTask" (explicit instructions for candidate), "expectedAnswer" (complete expected solution code/query/answer), "evaluationCriteria" (scoring guidelines), "correctAnswer" (same as expectedAnswer), "explanation".
   - Options must be null.
   - For SQL scenario questions (where subject is "SQL"), you MUST also generate:
     - "databaseSchema": array of SQL CREATE TABLE DDL statements
     - "sampleData": array of SQL INSERT INTO statements
7. Generate a complete set of high-quality assessment questions (between 15 and 30 questions total).

Response Schema:
{{
  "questions": [
    {{
      "subject": "Python",
      "topic": "Variables & Types",
      "type": "MCQ",
      "difficulty": "Easy",
      "question": "Which of the following is a mutable data type in Python?",
      "options": ["tuple", "str", "list", "int"],
      "correctAnswer": "list",
      "explanation": "Lists in Python are mutable, meaning their elements can be modified in place."
    }},
    {{
      "subject": "SQL",
      "topic": "Window Functions",
      "type": "SCENARIO",
      "difficulty": "Hard",
      "scenario": "Finding duplicate revenue records in financial audit database.",
      "question": "Write an SQL query using DENSE_RANK() to identify duplicate transaction records.",
      "problemStatement": "In an e-commerce transactions table, write a query to identify customer IDs with duplicate payments.",
      "candidateTask": "Write a query returning customer_id and transaction_count.",
      "expectedAnswer": "SELECT customer_id, COUNT(*) FROM transactions GROUP BY customer_id HAVING COUNT(*) > 1;",
      "evaluationCriteria": "Correct group by clause, having count, and valid SQL syntax.",
      "correctAnswer": "SELECT customer_id, COUNT(*) FROM transactions GROUP BY customer_id HAVING COUNT(*) > 1;",
      "explanation": "GROUP BY with HAVING COUNT(*) > 1 filters for groups with multiple transactions.",
      "databaseSchema": ["CREATE TABLE transactions (id INT, customer_id INT, amount DECIMAL(10,2));"],
      "sampleData": ["INSERT INTO transactions VALUES (1, 101, 50.00), (2, 101, 50.00);"]
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
2. Compare the Candidate Answer with the Recruiter's Correct Answer. Do NOT perform only an exact text match; evaluate semantic similarity. Accept answers that have the same meaning even if the wording is different.
3. Consider:
   - Correct concepts
   - Technical accuracy
   - Completeness
   - Relevance
   - Missing important points
4. Grade the answer on a scale of 0 to 100 in "score".
5. Set "similarity_score" as the semantic similarity percentage (0-100%).
6. "status" must be exactly one of: "Correct" (for similarity_score >= 80), "Partially Correct" (for similarity_score between 40 and 79), or "Incorrect" (for similarity_score < 40).
7. "ai_explanation" must be a constructive evaluation feedback and explanation.
8. "strengths" must describe what the candidate did well.
9. "missing_points" must list any missing important points (use "None" if there are none).
10. "suggested_improvement" must provide actionable suggestions on how to improve.
11. The JSON structure must match the schema below exactly.

Response Schema:
{{
  "similarity_score": 96,
  "score": 96,
  "status": "Correct",
  "ai_explanation": "The candidate understands the concept correctly. The explanation matches the expected answer.",
  "strengths": "Understands the concept of Virtual DOM and updates.",
  "missing_points": "None",
  "suggested_improvement": "Could mention how Virtual DOM minimizes direct browser layout recalculations."
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
                "similarity_score": 0,
                "status": "Incorrect",
                "ai_explanation": "AI evaluation failed to parse response.",
                "strengths": "None",
                "missing_points": "Could not determine missing points.",
                "suggested_improvement": "None"
            }

        return {
            "score": data.get("score", 0),
            "similarity_score": data.get("similarity_score", data.get("score", 0)),
            "status": data.get("status", "Incorrect"),
            "ai_explanation": data.get("ai_explanation", "No explanation provided."),
            "strengths": data.get("strengths", "No strengths highlighted."),
            "missing_points": data.get("missing_points", "No missing points highlighted."),
            "suggested_improvement": data.get("suggested_improvement", "No improvement areas identified.")
        }

    async def generate_overall_evaluation(
        self,
        assessment_name: str,
        total_questions: int,
        correct_count: int,
        partial_count: int,
        incorrect_count: int,
        final_percentage: float,
        questions_summary: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates overall assessment results feedback, strengths, weaknesses, and hiring recommendation.
        """
        questions_str = ""
        for i, q in enumerate(questions_summary):
            questions_str += f"""
Question {i+1}: {q.get('question')}
Candidate Answer: {q.get('candidate_answer')}
Correct Answer: {q.get('correct_answer')}
AI Evaluation Status: {q.get('status')}
Score: {q.get('score')}/100
"""

        prompt = f"""Generate an overall evaluation summary, strengths, weaknesses, and hiring recommendation for the candidate's completed assessment.

Assessment Name: {assessment_name}
Total Questions: {total_questions}
Correct: {correct_count}
Partially Correct: {partial_count}
Incorrect: {incorrect_count}
Final Score: {final_percentage}%

Details of each question:
{questions_str}

CRITICAL RULES AND CONSTRAINTS (YOU MUST COMPLY WITH ALL RULES):
1. Return ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks like ```json or ```. Return the raw JSON string directly.
2. Provide a cohesive, professional hiring manager summary in "overall_feedback".
3. Identify 2-3 key technical strengths in "overall_strengths".
4. Identify 1-2 areas of weakness or missing knowledge in "overall_weaknesses".
5. Provide a clear hiring recommendation in "hiring_recommendation" (e.g. "Recommended for Interview", "Partially Recommended", "Not Recommended").
6. The JSON structure must match the schema below exactly.

Response Schema:
{{
  "overall_feedback": "Cohesive summary of candidate's performance across all questions.",
  "overall_strengths": "List or paragraph describing candidate's technical strengths.",
  "overall_weaknesses": "List or paragraph describing candidate's weak areas.",
  "hiring_recommendation": "Recommended for Interview"
}}"""

        system_message = (
            "You are an expert technical recruiter and interviewer. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks (no ```json or ```). Your response must be clean JSON."
        )

        try:
            raw_response = await self.client.generate_chat_completion(prompt, system_message)
            cleaned_json = self._clean_json(raw_response)
            data = json.loads(cleaned_json)
            return {
                "overall_feedback": data.get("overall_feedback", "No overall feedback provided."),
                "overall_strengths": data.get("overall_strengths", "No overall strengths identified."),
                "overall_weaknesses": data.get("overall_weaknesses", "No overall weaknesses identified."),
                "hiring_recommendation": data.get("hiring_recommendation", "Recommended for Interview")
            }
        except Exception as e:
            logger.error(f"Failed to generate overall evaluation: {e}")
            return {
                "overall_feedback": "Successfully completed assessment.",
                "overall_strengths": "N/A",
                "overall_weaknesses": "N/A",
                "hiring_recommendation": "Recommended for Interview" if final_percentage >= 50.0 else "Not Recommended"
            }


    async def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyzes the candidate's resume text using Azure OpenAI.
        """
        prompt = f"""Analyze the candidate's resume text:

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

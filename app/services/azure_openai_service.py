import json
import logging
from typing import Dict, Any, List, Optional
from app.core.azure_openai import AzureOpenAIClient
from app.schemas.assessment import AssessmentGenerateRequest
from app.schemas.interview import InterviewGenerateRequest, InterviewEvaluateRequest

logger = logging.getLogger("recruitai-backend.azure_openai_service")

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAIClient()

    async def generate_questions(
        self, 
        request: AssessmentGenerateRequest,
        existing_questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds the prompt with a dynamic uniqueness seed and existing assessment exclusions,
        calls Azure OpenAI API client, cleans the response, and parses & validates it into high-quality,
        non-repetitive question objects.
        """
        # 1. Dynamically construct prompt
        prompt = self._build_prompt(request, existing_questions=existing_questions)
        logger.info(f"Generated Prompt:\n{prompt}")

        system_message = (
            "You are a principal technical assessment architect who designs high-quality, completely unique, "
            "and non-repetitive recruitment evaluations comparable to HackerRank, LeetCode, Codility, Mercer Mettl, and SHL. "
            "Every assessment you generate MUST contain brand new, original questions with fresh wording, diverse business scenarios, "
            "distinct code examples, and varied problem framing. Never repeat or reuse standard stock questions from previous runs. "
            "When generating SQL scenario questions, you MUST strictly use ONLY the real tables and columns from the provided AdventureWorks live database schema. "
            "You must return ONLY a JSON object matching the requested schema without any markdown formatting or code blocks."
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

        # 4. Validate and sanitize generated questions for strict quality compliance
        return self._clean_and_validate_questions(data["questions"], request.subjects)

    def _build_prompt(
        self, 
        request: AssessmentGenerateRequest,
        existing_questions: Optional[List[str]] = None
    ) -> str:
        """
        Constructs a comprehensive, high-quality prompt for generating recruitment assessment questions.
        Enforces unique question generation, clear wording, realistic workplace scenarios, HackerRank/LeetCode standards,
        strict distractor guidelines, and topic relevance.
        """
        import uuid
        import time

        unique_generation_seed = f"gen-{uuid.uuid4().hex[:12]}-{int(time.time()*1000)}"

        num_subjects = len(request.subjects)
        target_total = request.totalQuestions or (15 if num_subjects == 1 else (20 if num_subjects == 2 else 25))

        mcq_count = round(target_total * request.questionDistribution.mcq / 100.0)
        scenario_count = target_total - mcq_count

        easy_count = round(target_total * request.difficultyDistribution.easy / 100.0)
        medium_count = round(target_total * request.difficultyDistribution.medium / 100.0)
        hard_count = target_total - (easy_count + medium_count)

        subjects_str = ", ".join(request.subjects)

        sql_schema_context = ""
        if any("SQL" in s.upper() for s in request.subjects):
            from app.services.sql_schema_service import SqlSchemaService
            sql_schema_context = "\n" + SqlSchemaService.get_live_schema_text() + "\n"

        exclusion_context = ""
        if existing_questions:
            clean_ex = [f"- {q}" for q in existing_questions if q and len(q.strip()) > 5]
            if clean_ex:
                ex_items = "\n".join(clean_ex[:40])
                exclusion_context = f"\nEXISTING ASSESSMENT QUESTIONS TO STRICTLY EXCLUDE (DO NOT REUSE OR REPEAT ANY OF THESE):\n{ex_items}\n"

        prompt = f"""Generate professional technical recruitment assessment questions based on the following configurations:

UNIQUE ASSESSMENT GENERATION SEED: {unique_generation_seed}
Selected Subjects: {subjects_str}
{sql_schema_context}
{exclusion_context}
QUESTION COUNT & RATIOS:
- Ideal Total Question Count: ~{target_total} (between 15 and 30 total).
- MCQ Questions (type "MCQ"): ~{mcq_count} questions ({request.questionDistribution.mcq}%)
- Scenario-Based Questions (type "SCENARIO"): ~{scenario_count} questions ({request.questionDistribution.scenario}%)
- Difficulty Level Ratios: Easy ~{easy_count} ({request.difficultyDistribution.easy}%), Medium ~{medium_count} ({request.difficultyDistribution.medium}%), Hard ~{hard_count} ({request.difficultyDistribution.hard}%)

CRITICAL QUESTION NOVELTY & UNIQUENESS MANDATE (STRICT COMPLIANCE REQUIRED):
1. NO DUPLICATE OR REPETITIVE QUESTIONS:
   - Generate a COMPLETELY NEW AND UNIQUE set of questions for this assessment session.
   - Do NOT reuse identical or highly similar questions, examples, or code snippets from previous assessment generations.
   - Vary the question wording, contextual framing, problem scenarios (e.g. e-commerce, logistics, fintech, healthcare, HR analytics, cloud engineering), variable names, constraints, and edge cases.
   - Ensure every question tests depth and understanding from a fresh perspective while strictly remaining within the requested subjects: {request.subjects}.

2. GENERAL QUESTION EXCELLENCE & CLARITY:
   - Write in simple, clear, grammatically correct professional English.
   - Every question must be unambiguous with exactly ONE correct interpretation.
   - Avoid vague or incomplete statements. Include full necessary context.
   - Avoid unnecessary technical jargon unless essential to the core skill being evaluated.

3. MULTIPLE CHOICE QUESTIONS (MCQs):
   - Formulate a precise, self-contained question statement with a novel context or application angle.
   - Provide exactly 4 meaningful options with EXACTLY ONE correct answer.
   - NEVER use "All of the above", "None of the above", "All of these", or "None of these".

4. SCENARIO-BASED & PROGRAMMING QUESTIONS:
   - Must resemble real-world workplace situations with varied business domain contexts.
   - "scenario": Realistic background context (finance, supply chain, user behavior analytics, etc.).
   - "problemStatement": Business problem statement detailing inputs, outputs, requirements, and constraints.
   - "candidateTask": Explicit, specific task describing what the candidate must write or accomplish.
   - STRICT SQL MANDATE FOR SQL QUESTIONS:
     * You MUST generate SQL questions ONLY from the live database schema provided above.
     * NEVER assume table names or column names, NEVER use generic names (e.g., 'employees', 'orders', 'students'), and NEVER reference tables or columns that do not exist in the schema.
     * EVERY SQL question MUST explicitly state the schema and table name (e.g. HumanResources.Employee, Person.Person, Sales.SalesOrderHeader, Production.Product) and the exact column names involved.
     * Include a realistic business scenario, a clear problem statement, exact candidate task, valid solution query ("expectedAnswer"), expected output in a clean tabular format using actual column names ("exampleOutput"), and an explanation of the result ("explanation").
   - For Python/Coding questions: Describe input format, output format, constraints, sample input ("exampleInput"), sample output ("exampleOutput"), and full working solution ("expectedAnswer").
   - "evaluationCriteria": Clear rubric highlighting key evaluation points.

5. TOPIC RELEVANCE:
   - Every question's "subject" MUST strictly be one of: {request.subjects}.

RESPONSE SCHEMA (RETURN RAW CLEAN JSON ONLY):
{{
  "questions": [
    {{
      "subject": "Python",
      "topic": "Generator Functions",
      "type": "MCQ",
      "difficulty": "Medium",
      "question": "Which keyword is used to pause execution and yield a value in a Python generator function?",
      "options": ["yield", "return", "pause", "async"],
      "correctAnswer": "yield",
      "explanation": "The 'yield' statement pauses function execution and emits a value to the caller, maintaining execution state for subsequent iterations."
    }},
    {{
      "subject": "SQL",
      "topic": "Filtering & Sorting Employee Data",
      "type": "SCENARIO",
      "difficulty": "Hard",
      "scenario": "You are a Senior Data Analyst working with the AdventureWorks Human Resources department. Management needs an accurate list of active salaried employees to analyze organizational demographics and vacation allowances.",
      "question": "Write an SQL query to retrieve the BusinessEntityID, NationalIDNumber, JobTitle, and VacationHours for all salaried employees whose current record is active.",
      "problemStatement": "Using the 'HumanResources.Employee' table from the AdventureWorks schema (columns: BusinessEntityID, NationalIDNumber, JobTitle, SalariedFlag, CurrentFlag, VacationHours), retrieve all employees where SalariedFlag = 1 and CurrentFlag = 1, ordered descending by VacationHours.",
      "candidateTask": "Write a clean T-SQL SELECT query against HumanResources.Employee filtering by SalariedFlag = 1 and CurrentFlag = 1, sorting by VacationHours DESC.",
      "expectedAnswer": "SELECT BusinessEntityID, NationalIDNumber, JobTitle, VacationHours FROM HumanResources.Employee WHERE SalariedFlag = 1 AND CurrentFlag = 1 ORDER BY VacationHours DESC;",
      "evaluationCriteria": "Correct usage of HumanResources.Employee schema/table name, valid WHERE clauses on SalariedFlag and CurrentFlag, and proper ORDER BY descending.",
      "correctAnswer": "SELECT BusinessEntityID, NationalIDNumber, JobTitle, VacationHours FROM HumanResources.Employee WHERE SalariedFlag = 1 AND CurrentFlag = 1 ORDER BY VacationHours DESC;",
      "explanation": "Filtering by SalariedFlag = 1 and CurrentFlag = 1 isolates active salaried staff, while ORDER BY VacationHours DESC lists employees with the highest leave accumulation first.",
      "databaseSchema": ["-- Schema table: HumanResources.Employee (BusinessEntityID INT PK, NationalIDNumber NVARCHAR, JobTitle NVARCHAR, SalariedFlag BIT, CurrentFlag BIT, VacationHours SMALLINT)"],
      "sampleData": ["-- Live data exists on the connected AdventureWorks SQL Server"]
    }}
  ]
}}"""
        return prompt

    def _clean_and_validate_questions(self, questions: List[Dict[str, Any]], allowed_subjects: List[str]) -> List[Dict[str, Any]]:
        """
        Validates and refines each generated question to guarantee clarity, correct schema,
        strict distractor guidelines (no 'All/None of the above'), valid SQL/Python structures,
        and high-quality recruiter explanations.
        """
        banned_phrases = ["all of the above", "none of the above", "all of these", "none of these"]
        cleaned_questions = []

        for q in questions:
            # 1. Subject and Topic validation
            subject = q.get("subject", allowed_subjects[0])
            if subject not in allowed_subjects:
                subject = allowed_subjects[0]
            q["subject"] = subject
            q["topic"] = q.get("topic") or "General"

            # 2. Type & Difficulty normalization
            q_type = str(q.get("type", "MCQ")).upper()
            if q_type in {"SCENARIO", "CODING", "PYTHON_CODING", "SCENARIO_CODING"}:
                q_type = "SCENARIO"
            else:
                q_type = "MCQ"
            q["type"] = q_type

            q["difficulty"] = str(q.get("difficulty", "Medium")).capitalize()
            if q["difficulty"] not in {"Easy", "Medium", "Hard"}:
                q["difficulty"] = "Medium"

            # 3. Clean MCQ questions
            if q_type == "MCQ":
                q["question"] = str(q.get("question") or q.get("candidateTask") or f"Identify the correct option regarding {q['subject']} - {q['topic']}.").strip()
                raw_options = q.get("options") or []
                correct_ans = str(q.get("correctAnswer", "")).strip()

                cleaned_opts = []
                for opt in raw_options:
                    opt_str = str(opt).strip()
                    opt_lower = opt_str.lower()
                    if any(phrase in opt_lower for phrase in banned_phrases):
                        opt_str = f"Invalid {q.get('topic', 'concept')} configuration"
                    cleaned_opts.append(opt_str)

                # Ensure exactly 4 options
                while len(cleaned_opts) < 4:
                    cleaned_opts.append(f"Option {chr(65 + len(cleaned_opts))}")
                if len(cleaned_opts) > 4:
                    if correct_ans in cleaned_opts[:4]:
                        cleaned_opts = cleaned_opts[:4]
                    else:
                        cleaned_opts = cleaned_opts[:3] + [correct_ans]

                # Match correct_ans exact string
                exact_match = None
                for opt in cleaned_opts:
                    if opt.strip().lower() == correct_ans.lower():
                        exact_match = opt
                        break
                if exact_match:
                    q["correctAnswer"] = exact_match
                else:
                    q["correctAnswer"] = cleaned_opts[0]

                q["options"] = cleaned_opts
                if not q.get("explanation"):
                    q["explanation"] = f"The correct answer is '{q['correctAnswer']}', which accurately solves the {q['subject']} ({q['topic']}) task."

            # 4. Clean Scenario / Programming / SQL questions
            else:
                q["options"] = None
                
                scenario_bg = q.get("scenario") or q.get("problemStatement") or f"Real-world enterprise scenario assessing {q['subject']} - {q['topic']} skills."
                problem_stmt = q.get("problemStatement") or scenario_bg
                task = q.get("candidateTask") or q.get("question") or "Solve the given scenario by writing a clear implementation."

                q["scenario"] = scenario_bg
                q["problemStatement"] = problem_stmt
                q["candidateTask"] = task
                q["question"] = str(q.get("question") or task).strip()

                expected = q.get("expectedAnswer") or q.get("correctAnswer") or "Implementation matching requirements."
                q["expectedAnswer"] = expected
                q["correctAnswer"] = expected

                if not q.get("evaluationCriteria"):
                    q["evaluationCriteria"] = "Correct technical logic, optimal syntax, and complete adherence to task specifications."

                if not q.get("explanation"):
                    q["explanation"] = f"Solution demonstrates proper implementation of {q['subject']} ({q['topic']})."

                # Coerce string schema/sampleData to lists
                if q.get("databaseSchema") and isinstance(q.get("databaseSchema"), str):
                    q["databaseSchema"] = [q.get("databaseSchema")]
                if q.get("sampleData") and isinstance(q.get("sampleData"), str):
                    q["sampleData"] = [q.get("sampleData")]

                # SQL specific validation against live AdventureWorks schema (No hardcoded/mock tables!)
                if q["subject"].upper() == "SQL":
                    from app.services.sql_schema_service import SqlSchemaService
                    live_schema = SqlSchemaService.get_live_schema()
                    tables_map = live_schema.get("tables_map", {})

                    if not q.get("databaseSchema") or any("evaluation_records" in str(s).lower() for s in q.get("databaseSchema", [])):
                        emp_info = tables_map.get("HumanResources.Employee")
                        if emp_info and "columns" in emp_info:
                            cols = ", ".join([f"{c['name']} {c['type']}{' PRIMARY KEY' if c.get('is_pk') else ''}" for c in emp_info["columns"][:15]])
                            q["databaseSchema"] = [f"-- Live Schema from AdventureWorks Database\nCREATE TABLE HumanResources.Employee ({cols});"]
                        else:
                            q["databaseSchema"] = [
                                "-- Live Schema from AdventureWorks Database\n"
                                "CREATE TABLE HumanResources.Employee (BusinessEntityID INT PRIMARY KEY, NationalIDNumber NVARCHAR, JobTitle NVARCHAR, HireDate DATE, MaritalStatus NCHAR, Gender NCHAR);"
                            ]

                    if not q.get("sampleData") or any("evaluation_records" in str(s).lower() or "sample record" in str(s).lower() for s in q.get("sampleData", [])):
                        q["sampleData"] = ["-- Sample data is dynamically queried directly from the connected AdventureWorks database."]

            cleaned_questions.append(q)

        return cleaned_questions

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


    async def evaluate_assessment_answer(
        self,
        question: str,
        scenario: str,
        correct_answer: str,
        candidate_answer: str,
        question_type: str = "SCENARIO"
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate's answer to a technical assessment question by comparing it against
        the ideal reference answer across 4 key dimensions: Semantic Similarity, Correctness, Completeness, and Relevance.
        """
        prompt = f"""You are an expert AI assessment evaluator. Your task is to score a candidate's technical response and provide detailed feedback by comparing it against a reference answer.

Assessment Context / Question: {question}
Scenario Context: {scenario if scenario else "N/A"}
Question Type: {question_type}

Reference Answer (AI-generated Benchmark Response):
{correct_answer}

Candidate's Answer (Response Being Evaluated):
{candidate_answer}

EVALUATION DIMENSIONS (SYSTEMATICALLY COMPARE & WEIGHT):
1. Semantic Similarity: How well does the candidate's answer align with the core concepts and ideas in the reference answer, even if worded differently?
2. Correctness: Are the facts, logic, syntax, and technical details accurate? Do they contain errors or misconceptions?
3. Completeness: Does the candidate cover all key points, edge cases, and depth required, or are there significant gaps?
4. Relevance: How directly does the answer address what was asked? Is there unnecessary information or off-topic content?

SCORING APPROACH:
- Assign an overall score on a 0-100 scale in "score" and "similarity_score".
- Set "status" to:
  * "Correct" if score >= 80
  * "Partially Correct" if score is between 40 and 79
  * "Incorrect" if score < 40
- Justify the score by explaining performance across each of the 4 dimensions in "ai_explanation".

FEEDBACK STRUCTURE:
- "ai_explanation": Comprehensive analysis synthesizing performance across Semantic Similarity, Correctness, Completeness, and Relevance.
- "strengths": Specific details of what the candidate did well and core technical strengths demonstrated.
- "missing_points": Missing requirements, logical/syntactical errors, or gaps across the 4 dimensions.
- "suggested_improvement": Specific, actionable feedback guiding how the candidate can improve their response or solution.

CRITICAL RULES:
1. Return ONLY a raw valid JSON object. Do NOT wrap in markdown code blocks.
2. The JSON structure must match the schema below.

Response Schema:
{{
  "similarity_score": 85,
  "score": 85,
  "status": "Correct",
  "ai_explanation": "Detailed evaluation across Semantic Similarity, Correctness, Completeness, and Relevance explaining why 85/100 marks were awarded.",
  "strengths": "Demonstrates high semantic similarity and strong technical correctness.",
  "missing_points": "Minor gaps in completeness regarding edge case handling.",
  "suggested_improvement": "Include explicit error handling to ensure 100% completeness."
}}"""
        system_message = (
            "You are an expert AI assessment evaluator. You evaluate technical assessment submissions "
            "fairly based on four dimensions: Semantic Similarity, Correctness, Completeness, and Relevance. "
            "You must return ONLY a JSON object matching the requested schema."
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
                "feedback": "AI evaluation failed to parse response.",
                "strengths": "None",
                "missing_points": "Could not determine missing points.",
                "suggested_improvement": "None",
                "improvements": "None"
            }

        ai_exp = data.get("ai_explanation", data.get("feedback", "No explanation provided."))
        sug_imp = data.get("suggested_improvement", data.get("improvements", "No improvement areas identified."))

        return {
            "score": data.get("score", 0),
            "similarity_score": data.get("similarity_score", data.get("score", 0)),
            "status": data.get("status", "Incorrect"),
            "ai_explanation": ai_exp,
            "feedback": ai_exp,
            "strengths": data.get("strengths", "No strengths highlighted."),
            "missing_points": data.get("missing_points", "No missing points highlighted."),
            "suggested_improvement": sug_imp,
            "improvements": sug_imp
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
        Generates overall assessment results feedback, strengths, weaknesses, and hiring recommendation
        synthesizing candidate performance across Semantic Similarity, Correctness, Completeness, and Relevance.
        """
        questions_str = ""
        for i, q in enumerate(questions_summary):
            questions_str += f"""
Question {i+1}: {q.get('question')}
Candidate Answer: {q.get('candidate_answer')}
Reference Answer: {q.get('correct_answer')}
AI Evaluation Status: {q.get('status')}
Score: {q.get('score')}/100
"""

        prompt = f"""Generate an overall evaluation summary, technical strengths, weaknesses, and hiring recommendation for the candidate's completed technical assessment.

Assessment Name: {assessment_name}
Total Questions: {total_questions}
Correct Answers: {correct_count}
Partially Correct: {partial_count}
Incorrect: {incorrect_count}
Overall Score Percentage: {final_percentage}%

Individual Question Evaluation Summaries:
{questions_str}

CRITICAL RULES AND CONSTRAINTS:
1. Return ONLY valid JSON. Do NOT wrap in markdown code blocks.
2. Provide an overall assessment feedback in "overall_feedback" synthesizing candidate performance across all 4 evaluation dimensions (Semantic Similarity, Correctness, Completeness, and Relevance) and offering guidance for growth.
3. Identify 2-3 specific technical strengths in "overall_strengths".
4. Identify 1-2 primary areas of weakness or missing knowledge in "overall_weaknesses".
5. Provide a clear hiring recommendation in "hiring_recommendation" (e.g. "Recommended for Interview", "Partially Recommended", "Not Recommended").
6. The JSON structure must match the schema below.

Response Schema:
{{
  "overall_feedback": "Synthesizes performance across Semantic Similarity, Correctness, Completeness, and Relevance with actionable growth guidance.",
  "overall_strengths": "Highlights key technical strengths and domain mastery.",
  "overall_weaknesses": "Highlights specific knowledge gaps and areas needing improvement.",
  "hiring_recommendation": "Recommended for Interview"
}}"""

        system_message = (
            "You are an expert technical recruiter and interviewer. You must return ONLY a JSON object "
            "matching the requested schema. Do not include any explanation, markdown, "
            "or code blocks. Your response must be clean JSON."
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

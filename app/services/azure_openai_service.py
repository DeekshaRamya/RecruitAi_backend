import json
import logging
import re
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

        has_aptitude = any("APTITUDE" in str(s).upper() for s in request.subjects)
        has_python = any("PYTHON" in str(s).upper() for s in request.subjects)
        has_sql = any("SQL" in str(s).upper() for s in request.subjects)

        system_message = (
            "You are an expert AI technical and aptitude assessment generator that produces high-quality placement exam questions for recruiters. "
            "Your task is to generate unique, precise assessment questions strictly conforming to the requested subjects, topics, distributions, and schema rules.\n\n"
            "APTITUDE ASSESSMENT MANDATE:\n"
            "- Standard placement topics ONLY: LCM, HCF, Average, Profit and Loss, Percentage, Ratio and Proportion, Simple Interest, Compound Interest, Time and Work, Time Speed and Distance, Pipes and Cisterns, Ages, Partnership, Mixture and Alligation, Number System, Divisibility, Simplification, Probability, Permutation and Combination, Data Interpretation, Series, Calendar, Clock, Blood Relations, Direction Sense, Coding-Decoding, Seating Arrangement, Logical Reasoning.\n"
            "- MCQ questions: Exactly 4 realistic options with 1 correct answer and a step-by-step calculation explanation.\n"
            "- Scenario-Based questions: Strictly NO options (`options`: null). Descriptive, calculation-oriented aptitude problems where candidates type the answer in a single input field (`placeholder`: 'Enter your answer' or 'Type your answer here', `answerType`: 'NUMBER' or 'TEXT'). Include expectedAnswer, correctAnswer, answerType, placeholder, and step-by-step calculation explanation.\n"
            "- Difficulty calibration: Easy (formula-based), Medium (2+ calculation steps), Hard (complex multi-step reasoning).\n\n"
            "PYTHON & TECHNICAL ASSESSMENT MANDATE:\n"
            "- Python coding questions must be framed as real-world scenarios with beginner/intermediate concepts, dynamic starter code, and visible sample test case.\n"
            "- SQL questions must adhere strictly to AdventureWorks schema.\n"
            "- Every generated question MUST be completely unique and non-repetitive."
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
        Enforces unique question generation, clear wording, beginner/intermediate Python coding challenges,
        AdventureWorks SQL schema compliance, strict Aptitude placement topic compliance, distractor guidelines, and topic relevance.
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

        prompt = f"""Generate professional technical and aptitude recruitment assessment questions based on the following configurations:

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
   - Vary the question wording, variable names, constraints, and edge cases while staying strictly within the selected topics.
   - Ensure every question tests depth and understanding from a fresh perspective while strictly remaining within the requested subjects: {request.subjects}.

2. GENERAL QUESTION EXCELLENCE & CLARITY:
   - Write in simple, clear, grammatically correct professional English.
   - Every question must be unambiguous with exactly ONE correct implementation/solution.
   - Avoid vague or incomplete statements. Include full necessary context.

3. MULTIPLE CHOICE QUESTIONS (MCQs):
   - Formulate a precise, self-contained question statement.
   - Provide exactly 4 meaningful options with EXACTLY ONE correct answer.
   - NEVER use "All of the above", "None of the above", "All of these", or "None of these".

4. APTITUDE ASSESSMENT MANDATE (STRICT COMPLIANCE FOR SUBJECT "Aptitude"):
   - TOPICS: Questions MUST be generated ONLY from standard placement topics: LCM, HCF, Average, Profit and Loss, Percentage, Ratio and Proportion, Simple Interest, Compound Interest, Time and Work, Time Speed and Distance, Pipes and Cisterns, Ages, Partnership, Mixture and Alligation, Number System, Divisibility, Simplification, Probability, Permutation and Combination, Data Interpretation, Series, Calendar, Clock, Blood Relations, Direction Sense, Coding-Decoding, Seating Arrangement, Logical Reasoning.
   - QUESTION TYPES:
     * MCQ Questions (`"type": "MCQ"`):
       - Formulate a clear placement-exam style question.
       - Provide EXACTLY FOUR realistic option choices in `"options"`.
       - Exactly ONE correct option choice in `"correctAnswer"`.
       - Detailed step-by-step mathematical or logical reasoning in `"explanation"`.
     * Scenario-Based Questions (`"type": "SCENARIO"`):
       - DO NOT generate options (`"options"` MUST BE null / omitted). NEVER generate MCQ options for Scenario-Based Aptitude questions under any circumstance.
       - ALWAYS generate complete, self-contained, and fully detailed questions. NEVER generate truncated or incomplete questions.
       - MUST follow this exact 4-part structure:
         1. `"scenario"`: Full real-world / business context containing all necessary numerical values and parameters. (e.g. "An employee deposits $12,000 in a savings scheme offering simple interest at 7% per annum for 3 years.")
         2. `"question"`: Explicit calculation task asking what the candidate must calculate. (e.g. "Calculate the simple interest earned after 3 years.")
         3. `"problemStatement"`: Full combined problem statement (Scenario + Task).
         4. `"placeholder"`: `"Enter your answer"`
       - MUST also include:
         * `"answerType"`: `"NUMBER"` or `"TEXT"`
         * `"expectedAnswer"` and `"correctAnswer"`: exact expected numerical/text result string (e.g. `"2520"`)
         * `"explanation"`: detailed step-by-step calculation or reasoning.
       - NEVER generate incomplete questions such as returning only a scenario description without the actual calculation task request.
   - DIFFICULTY CALIBRATION FOR APTITUDE:
     * Easy: Basic formula-based questions (single step calculations).
     * Medium: Requires two or more calculation steps.
     * Hard: Complex multi-step reasoning or multi-concept problems.
   - NOVELTY & VARIATION: Vary numbers, names, contexts, and wording on every generation so no two assessment questions are identical.

5. SCENARIO-BASED PYTHON CODING QUESTIONS:
   - STRICT MANDATE FOR REAL-WORLD SCENARIOS: Every Python coding question MUST be framed as a real-world scenario (e.g., School, College, Library, Hospital, Railway, Airport, Cricket, Restaurant, Shopping Mall, Parking, Employee Salary, Student Marks, Banking, Delivery, Weather, Attendance, Electricity Bill, Mobile Recharge, Hotel, Supermarket, Inventory, Online Shopping, Movie Ticket Booking, Examination, Bus Reservation).
   - BAN ON PLAIN TEXTBOOK QUESTIONS: Never generate plain textbook questions like "Find the factorial" or "Check palindrome". Always wrap every question into a real-world scenario.
   - ALLOWED BEGINNER / INTERMEDIATE TOPICS: Generate ONLY from beginner or intermediate Python concepts.
   - CANDIDATE OUTPUT FORMAT: Candidates solve using print(). Do NOT expect return.
   - SAMPLE TEST CASE ONLY: Generate exactly ONE visible Sample Input (`sampleInput`) and Sample Output (`sampleOutput`).

6. TOPIC RELEVANCE:
   - Every question's "subject" MUST strictly be one of: {request.subjects}.

RESPONSE SCHEMA (RETURN RAW CLEAN JSON ONLY):
{{
  "questions": [
    {{
      "subject": "Aptitude",
      "topic": "Time Speed and Distance",
      "type": "MCQ",
      "difficulty": "Easy",
      "question": "A train 150 meters long is running at a speed of 54 km/hr. How much time will it take to cross a telegraph post?",
      "options": ["8 seconds", "10 seconds", "12 seconds", "15 seconds"],
      "correctAnswer": "10 seconds",
      "explanation": "Speed in m/s = 54 * (5/18) = 15 m/s. Time = Distance / Speed = 150 / 15 = 10 seconds."
    }},
    {{
      "subject": "Aptitude",
      "topic": "Profit and Loss",
      "type": "SCENARIO",
      "difficulty": "Medium",
      "scenario": "A merchant marks his goods 20% above the cost price and allows a discount of 10% on the marked price for cash payment.",
      "question": "Calculate the merchant's net profit percentage.",
      "answerType": "NUMBER",
      "placeholder": "Enter your answer",
      "expectedAnswer": "8",
      "correctAnswer": "8",
      "options": null,
      "explanation": "Let Cost Price = 100. Marked Price = 120. Selling Price = 120 - 10% of 120 = 108. Net Profit = 108 - 100 = 8%."
    }},
    {{
      "subject": "Python",
      "topic": "Strings",
      "type": "MCQ",
      "difficulty": "Easy",
      "question": "Which Python string method is used to convert all characters in a string to uppercase?",
      "options": ["upper()", "toUpper()", "uppercase()", "raise_case()"],
      "correctAnswer": "upper()",
      "explanation": "The upper() method returns a copy of the string converted to uppercase."
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
        banned_inputs = ["no input", "no input.", "n/a", "none", "null", "", "-", "tbd", "write python code", "solve the problem", "solve the given coding problem", "complete the function", "placeholder"]
        banned_outputs = ["no output", "no output.", "n/a", "none", "null", "", "-", "tbd", "placeholder"]

        def is_placeholder(val: str, banned_list: list) -> bool:
            clean_v = str(val or "").strip().lower()
            if clean_v in banned_list:
                return True
            if any(b in clean_v for b in ["no input", "no output", "n/a", "tbd"]):
                return True
            return False

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

    def _normalize_and_validate_python_question(self, q: dict) -> dict:
        """
        Validates and normalizes Python scenario questions to enforce 100% consistency
        between problem statement, function signature, parameter naming, input format,
        sample input, starter code, and visible test cases (Rules 1 - 7).
        """
        topic_lower = str(q.get("topic", "")).lower()
        title_lower = str(q.get("title", "")).lower()
        scen_raw = str(q.get("scenario") or q.get("problemStatement") or q.get("question") or "").strip()
        q_full_text = f"{title_lower} {topic_lower} {scen_raw}".lower()

        # Extract raw sample input & output
        vtc_dict = q.get("visibleTestCase")
        if not isinstance(vtc_dict, dict):
            visible_list = q.get("visibleTestCases") or []
            if isinstance(visible_list, list) and len(visible_list) > 0 and isinstance(visible_list[0], dict):
                vtc_dict = visible_list[0]
            else:
                vtc_dict = {}

        raw_in = str(vtc_dict.get("input") or q.get("sampleInput") or q.get("exampleInput") or "").strip()
        raw_out = str(vtc_dict.get("expectedOutput") if vtc_dict.get("expectedOutput") is not None else (vtc_dict.get("output") or q.get("sampleOutput") or q.get("exampleOutput") or "")).strip()

        # 1. Determine logical parameter name & data type based on problem requirements (Rules 1, 5, 6)
        param_name = "numbers"
        param_type_desc = "numbers: List[int]"
        is_list = False
        is_matrix = False
        is_string = False
        is_multi = False
        multi_params = []
        multi_types = []

        if any(w in q_full_text for w in ["parcel weight", "package weight", "parcel", "weight"]):
            param_name = "weights"
            param_type_desc = "weights: List[int]"
            is_list = True
        elif any(w in q_full_text for w in ["price", "revenue", "sales", "store", "bookstore"]):
            if "tax" in q_full_text:
                is_multi = True
                multi_params = ["price", "tax"]
                multi_types = ["price: float", "tax: float"]
            else:
                param_name = "prices"
                param_type_desc = "prices: List[int]"
                is_list = True
        elif any(w in q_full_text for w in ["mark", "score", "student"]):
            param_name = "marks"
            param_type_desc = "marks: List[int]"
            is_list = True
        elif any(w in q_full_text for w in ["temperature", "weather"]):
            param_name = "temperatures"
            param_type_desc = "temperatures: List[int]"
            is_list = True
        elif any(w in q_full_text for w in ["employee", "salary"]):
            param_name = "employees"
            param_type_desc = "employees: List[int]"
            is_list = True
        elif any(w in q_full_text for w in ["vowel", "palindrome", "reverse string", "uppercase", "lowercase", "word", "sentence", "text", "string"]):
            if "merge string" in q_full_text or "two string" in q_full_text:
                is_multi = True
                multi_params = ["str1", "str2"]
                multi_types = ["str1: str", "str2: str"]
            else:
                param_name = "text"
                param_type_desc = "text: str"
                is_string = True
        elif any(w in q_full_text for w in ["matrix", "2d", "grid", "diagonal"]):
            param_name = "matrix"
            param_type_desc = "matrix: List[List[int]]"
            is_matrix = True
        elif "recharge" in q_full_text and "fee" in q_full_text:
            is_multi = True
            multi_params = ["recharge_amount", "service_fee"]
            multi_types = ["recharge_amount: float", "service_fee: float"]
        else:
            param_name = "numbers"
            param_type_desc = "numbers: List[int]"
            is_list = True

        # 2. Build exact function signature string (Rule 1 & Rule 6)
        if is_multi:
            func_sig = f"def solution({', '.join(multi_params)}):"
            input_fmt_str = "\n".join(multi_types)
        else:
            func_sig = f"def solution({param_name}):"
            input_fmt_str = param_type_desc

        # 3. Clean and convert raw console input to match Python parameter data structure (Rule 2, 4, 7)
        clean_sample_in = raw_in
        if is_list:
            lines = [l.strip() for l in raw_in.splitlines() if l.strip()]
            if len(lines) >= 2 and lines[0].lstrip('-').isdigit() and not lines[1].startswith('['):
                elements = []
                for line in lines[1:]:
                    elements.extend([x.strip(',') for x in line.split() if x.strip(',')])
                clean_sample_in = "[" + ", ".join(elements) + "]"
            elif len(lines) == 1 and not raw_in.startswith('['):
                tokens = [x.strip(',') for x in raw_in.replace(',', ' ').split() if x.strip(',')]
                if len(tokens) > 0:
                    clean_sample_in = "[" + ", ".join(tokens) + "]"
            elif not clean_sample_in or any(b in clean_sample_in.lower() for b in ["no input", "placeholder", "tbd"]):
                clean_sample_in = "[12, 18, 9, 25, 17]" if param_name == "weights" else ("[120, 80, 50]" if param_name == "prices" else "[10, 20, 30]")
        elif is_string:
            if not clean_sample_in or clean_sample_in.startswith("[") or any(b in clean_sample_in.lower() for b in ["no input", "placeholder", "tbd"]):
                clean_sample_in = '"hello"'
            elif not clean_sample_in.startswith('"') and not clean_sample_in.startswith("'"):
                clean_sample_in = f'"{clean_sample_in}"'
        elif is_matrix:
            if not clean_sample_in.startswith("["):
                clean_sample_in = "[[1, 2], [3, 4]]"

        if not raw_out or any(b in raw_out.lower() for b in ["no output", "placeholder", "tbd"]):
            raw_out = "25" if is_list else ("True" if is_string else "100")

        # Derive exact, explicit Output Format return type (banning vague/generic statements)
        output_fmt_str = self._infer_explicit_output_format(q_full_text, raw_out)

        # 4. Construct formatted question text
        clean_scenario = scen_raw
        for p in ["Scenario:", "Task:", "Input Format:", "Output Format:", "Example:", "Input:", "Output:"]:
            clean_scenario = re.sub(rf'^{p}\s*', '', clean_scenario, flags=re.IGNORECASE).strip()

        formatted_q = (
            f"Scenario:\n"
            f"{clean_scenario}\n\n"
            f"Task:\n"
            f"Write a Python function:\n\n"
            f"{func_sig}\n\n"
            f"that takes the specified inputs and returns the expected result.\n\n"
            f"Input Format:\n"
            f"{input_fmt_str}\n\n"
            f"Output Format:\n"
            f"{output_fmt_str}\n\n"
            f"Example:\n\n"
            f"Input:\n"
            f"{clean_sample_in}\n\n"
            f"Output:\n"
            f"{raw_out}"
        )

        starter_code = f"{func_sig}\n    pass"

        q["functionSignature"] = func_sig
        q["starterCode"] = starter_code
        q["starter_code"] = starter_code
        q["inputFormat"] = input_fmt_str
        q["outputFormat"] = output_fmt_str
        q["scenario"] = clean_scenario
        q["problemStatement"] = formatted_q
        q["question"] = formatted_q
        q["sampleInput"] = clean_sample_in
        q["sampleOutput"] = raw_out
        q["exampleInput"] = clean_sample_in
        q["exampleOutput"] = raw_out

        visible_tc = {"input": clean_sample_in, "expectedOutput": raw_out, "output": raw_out}
        q["visibleTestCase"] = visible_tc
        q["visibleTestCases"] = [visible_tc]
        q["hiddenTestCases"] = []
        q["testCases"] = {
            "visible": [{"input": clean_sample_in, "output": raw_out}],
            "hidden": []
        }

        return q

    def _infer_explicit_output_format(self, q_full_text: str, raw_out: str) -> str:
        """
        Derives explicit, exact return type descriptions for Python questions,
        strictly prohibiting generic statements like 'Return the computed result.'
        Allowed formats:
        - 'Return an integer.'
        - 'Return a string.'
        - 'Return a boolean.'
        - 'Return a float.'
        - 'Return a List[int].'
        - 'Return a List[str].'
        - 'Return a Dictionary.'
        """
        clean_out = str(raw_out or "").strip()
        clean_text = q_full_text.lower()

        # 1. Inspect raw output structure if present
        if clean_out.startswith("{") and clean_out.endswith("}"):
            return "Return a Dictionary."

        if clean_out.startswith("[") and clean_out.endswith("]"):
            try:
                parsed = json.loads(clean_out.replace("'", '"'))
                if isinstance(parsed, list):
                    if len(parsed) > 0 and isinstance(parsed[0], str):
                        return "Return a List[str]."
                    return "Return a List[int]."
            except Exception:
                pass
            if '"' in clean_out or "'" in clean_out or any(c.isalpha() for c in clean_out):
                return "Return a List[str]."
            return "Return a List[int]."

        if clean_out.lower() in ["true", "false"]:
            return "Return a boolean."

        if "." in clean_out and clean_out.replace(".", "", 1).lstrip("-").isdigit():
            return "Return a float."

        if clean_out.lstrip("-").isdigit():
            return "Return an integer."

        # 2. Inspect problem keywords if raw output is string/text or ambiguous
        if any(w in clean_text for w in ["palindrome", "check if", "is_valid", "boolean", "true or false", "is valid"]):
            return "Return a boolean."

        if any(w in clean_text for w in ["list of integer", "array of integer", "return list", "all evens", "filter"]):
            return "Return a List[int]."

        if any(w in clean_text for w in ["list of string", "array of string", "words list", "split"]):
            return "Return a List[str]."

        if any(w in clean_text for w in ["dictionary", "hash map", "mapping", "counts dict", "word frequency"]):
            return "Return a Dictionary."

        if any(w in clean_text for w in ["average", "float", "percentage", "ratio", "tax", "fee", "rate"]):
            return "Return a float."

        if any(w in clean_text for w in ["vowel", "reverse string", "text", "sentence", "name", "word", "concat", "merge string"]):
            return "Return a string."

        # Default to integer if count, sum, max, min, total, length, weight, price, mark
        if any(w in clean_text for w in ["sum", "total", "max", "maximum", "min", "minimum", "count", "weight", "price", "mark", "salary", "length"]):
            return "Return an integer."

        # Fallback to string if raw_out has non-digit text
        if any(c.isalpha() for c in clean_out):
            return "Return a string."

        return "Return an integer."

    def _generate_python_starter_and_sig(self, q: dict, s_in: str) -> tuple[str, str]:
        """
        Delegates starter code and function signature generation to the comprehensive starter code generator.
        """
        from app.utils.starter_code_generator import generate_python_starter_code
        dynamic_starter = generate_python_starter_code(q, override_sample_input=s_in)
        sig_match = re.search(r"def\s+(\w+\s*\(.*?\))", dynamic_starter)
        sig_name = sig_match.group(1) if sig_match else "solution(numbers)"
        return sig_name, dynamic_starter

    def _derive_aptitude_output_format(self, raw_expected: str, topic: str, existing_fmt: Optional[str] = None) -> tuple[str, str]:
        """
        Derives the 4-part Output Format guidelines and cleans expectedAnswer.
        Output formats:
        - Percentage
        - Decimal
        - Integer
        - Currency
        - Time
        - Text
        """
        import re
        topic_lower = (topic or "").lower()
        exp_str = (raw_expected or "").strip()
        
        is_pct = "%" in exp_str or any(kw in topic_lower for kw in ["percentage", "interest", "profit and loss", "discount", "margin", "probability"])
        is_curr = any(sym in exp_str for sym in ["$", "₹", "€", "£"]) or any(kw in topic_lower for kw in ["cost", "price", "salary", "partnership", "investment", "amount"])
        is_time = any(unit in exp_str.lower() for unit in ["day", "hour", "minute", "second", "year", "month"]) or any(kw in topic_lower for kw in ["work", "speed", "distance", "cistern", "pipe", "clock", "calendar", "age"])

        # 1. Percentage
        if "%" in exp_str or (is_pct and not is_curr and not is_time and any(c.isdigit() for c in exp_str)):
            clean_exp = re.sub(r'[\$,₹,€,%,]', '', exp_str).strip()
            try:
                val = float(clean_exp)
                clean_exp = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
            except ValueError:
                pass
            fmt = (
                "Output Format:\n"
                "- Enter only the numeric value.\n"
                "- Do NOT include the '%' symbol.\n"
                "- Round your answer to exactly 2 decimal places if required.\n"
                "Example: 25 or 25.50"
            )
            return fmt, clean_exp

        # 2. Currency
        if any(sym in exp_str for sym in ["$", "₹", "€", "£"]) or (is_curr and any(c.isdigit() for c in exp_str)):
            clean_exp = re.sub(r'[\$,₹,€,£,]', '', exp_str).strip()
            try:
                val = float(clean_exp)
                clean_exp = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
            except ValueError:
                pass
            fmt = (
                "Output Format:\n"
                "- Enter only the numeric amount.\n"
                "- Do not include currency symbols such as ₹, $, €, etc.\n"
                "- Round to 2 decimal places if required.\n"
                "Example: 1250 or 1250.50"
            )
            return fmt, clean_exp

        # 3. Time / Units
        if is_time and any(c.isdigit() for c in exp_str):
            clean_exp = exp_str.replace(",", "").strip()
            fmt = (
                "Output Format:\n"
                "- Enter only the numeric value followed by the required unit if explicitly requested in the question.\n"
                "Example: 5 days"
            )
            return fmt, clean_exp

        # 4. Decimal vs Integer (for purely numeric answers)
        if any(c.isdigit() for c in exp_str):
            clean_exp = re.sub(r'[^\d\.]', '', exp_str).strip()
            if "." in clean_exp:
                try:
                    val = float(clean_exp)
                    clean_exp = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
                except ValueError:
                    pass
                fmt = (
                    "Output Format:\n"
                    "- Enter only the numeric value.\n"
                    "- Round to exactly 2 decimal places unless otherwise specified.\n"
                    "Example: 12.75"
                )
                return fmt, clean_exp
            else:
                fmt = (
                    "Output Format:\n"
                    "- Enter only the whole number.\n"
                    "- Do not include commas, units, currency symbols, or additional text.\n"
                    "Example: 450"
                )
                return fmt, clean_exp

        # 5. Text
        fmt = (
            "Output Format:\n"
            "- Enter only the required word or phrase exactly as requested."
        )
        return fmt, exp_str

    def _clean_and_validate_questions(
        self, 
        questions: List[Dict[str, Any]], 
        target_subjects: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Post-processes generated questions to guarantee non-empty inputs/outputs,
        valid distractor options, AdventureWorks SQL schema compliance, and strictly formatted starter code.
        """
        import re

        cleaned_questions = []
        banned_inputs = {"no input", "n/a", "none", "null", "undefined", ""}
        banned_outputs = {"no output", "n/a", "none", "null", "undefined", ""}

        def is_placeholder(val: str, banned_set: set) -> bool:
            if not val:
                return True
            clean_v = val.strip().lower()
            return clean_v in banned_set

        for idx, q in enumerate(questions):
            q["id"] = q.get("id") or f"q_{idx+1}"
            q["subject"] = str(q.get("subject") or (target_subjects[0] if target_subjects else "General")).strip()
            q["type"] = str(q.get("type") or "MCQ").strip().upper()
            q["difficulty"] = str(q.get("difficulty") or "Medium").strip().capitalize()
            q["topic"] = str(q.get("topic") or "General").strip()

            # 1. Standardize subject capitalization
            if "SQL" in q["subject"].upper():
                q["subject"] = "SQL"
            elif "PYTHON" in q["subject"].upper():
                q["subject"] = "Python"
            elif any(k in q["subject"].upper() for k in ["APTITUDE", "QUANT", "LOGICAL", "REASONING"]):
                q["subject"] = "Aptitude"

            # 2. Fix question type if mismatched
            if q["type"] in ["CODING", "PYTHON_CODING", "SCENARIO_CODING"]:
                q["type"] = "SCENARIO"

            # 3. Clean MCQ questions
            if q["type"] == "MCQ":
                opts = q.get("options")
                if not isinstance(opts, list) or len(opts) < 4:
                    q["options"] = ["Option A", "Option B", "Option C", "Option D"]
                else:
                    clean_opts = [str(opt).strip() for opt in opts[:4]]
                    while len(clean_opts) < 4:
                        clean_opts.append(f"Option {chr(65+len(clean_opts))}")
                    q["options"] = clean_opts

                if not q.get("correctAnswer") or is_placeholder(str(q.get("correctAnswer")), banned_outputs):
                    q["correctAnswer"] = q["options"][0]
                
                if not q.get("explanation") or is_placeholder(str(q.get("explanation")), banned_outputs):
                    q["explanation"] = f"The correct answer is '{q['correctAnswer']}', which accurately solves the {q['subject']} ({q['topic']}) task."

            # 4. Clean Scenario / Programming / SQL / Aptitude questions
            else:
                q["options"] = None
                
                if q["subject"].upper() == "APTITUDE":
                    q["answerType"] = str(q.get("answerType") or q.get("answer_type") or "NUMBER").upper()
                    if q["answerType"] not in ["NUMBER", "TEXT"]:
                        q["answerType"] = "NUMBER"
                    q["placeholder"] = str(q.get("placeholder") or "Enter your answer").strip()
                    if not q["placeholder"] or any(banned in q["placeholder"].lower() for banned in ["code", "sql", "write"]):
                        q["placeholder"] = "Enter your answer"

                    raw_expected = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()
                    if not raw_expected:
                        raw_expected = "0"
                    raw_topic = str(q.get("topic") or "Aptitude").strip()

                    # Derive output format and clean expected answer
                    output_fmt, clean_expected = self._derive_aptitude_output_format(raw_expected, raw_topic, q.get("outputFormat"))
                    q["outputFormat"] = output_fmt
                    q["expectedAnswer"] = clean_expected
                    q["correctAnswer"] = clean_expected

                    # Extract distinct scenario (context) and question (task)
                    raw_scen = str(q.get("scenario") or "").strip()
                    raw_q = str(q.get("question") or q.get("problemStatement") or q.get("candidateTask") or "").strip()

                    # Split scenario text into context and task if task is embedded
                    if ("Task:" in raw_scen or any(kw in raw_scen for kw in ["Calculate", "Find", "Determine", "What is", "How many"])) and (not raw_q or raw_q == raw_scen):
                        for delimiter in ["Task:", "Question:", "Calculate", "Find", "Determine", "What is", "How many"]:
                            if delimiter in raw_scen and raw_scen.index(delimiter) > 15:
                                split_idx = raw_scen.index(delimiter)
                                raw_q = raw_scen[split_idx:].strip()
                                if raw_q.startswith("Task:"):
                                    raw_q = raw_q[5:].strip()
                                raw_scen = raw_scen[:split_idx].strip()
                                break

                    if not raw_scen:
                        raw_scen = f"A real-world placement scenario involving {raw_topic} requiring precise calculation."
                    
                    if not raw_q or raw_q == raw_scen or len(raw_q) < 5:
                        raw_q = f"Calculate the exact numerical result for this {raw_topic} problem."

                    # Ensure question starts cleanly with a clear task command
                    if not any(raw_q.lower().startswith(kw) for kw in ["calculate", "find", "determine", "what", "how", "solve", "compute"]):
                        raw_q = f"Calculate {raw_q[0].lower() + raw_q[1:] if raw_q else 'the answer.'}"

                    q["scenario"] = raw_scen
                    q["question"] = raw_q
                    q["problemStatement"] = f"{raw_scen}\n\nTask: {raw_q}\n\n{output_fmt}"
                    q["candidateTask"] = raw_q

                    if not q.get("explanation") or is_placeholder(str(q.get("explanation")), banned_outputs):
                        q["explanation"] = f"Step-by-step mathematical/logical solution yielding expected answer: {clean_expected}."

                    # Remove programming and SQL specific fields for Aptitude Scenarios
                    q["starterCode"] = None
                    q["starter_code"] = None
                    q["databaseSchema"] = None
                    q["sampleData"] = None
                    q["functionSignature"] = None
                    q["visibleTestCase"] = None
                    q["visibleTestCases"] = None
                    q["hiddenTestCases"] = None

                elif q["subject"].upper() == "PYTHON":
                    q = self._normalize_and_validate_python_question(q)
                else:
                    scenario_bg = q.get("scenario") or q.get("problemStatement") or f"Real-world enterprise scenario assessing {q['subject']} - {q['topic']} skills."
                    sql_ex_in = str(q.get("exampleInput") or q.get("sampleInput") or "SalariedFlag = 1, CurrentFlag = 1").strip()
                    sql_ex_out = str(q.get("exampleOutput") or q.get("sampleOutput") or "List of active employee records ordered by VacationHours DESC").strip()
                    if is_placeholder(sql_ex_in, banned_inputs):
                        sql_ex_in = "SalariedFlag = 1, CurrentFlag = 1"
                    if is_placeholder(sql_ex_out, banned_outputs):
                        sql_ex_out = "List of active employee records ordered by VacationHours DESC"
                    q["exampleInput"] = sql_ex_in
                    q["exampleOutput"] = sql_ex_out
                    q["sampleInput"] = sql_ex_in
                    q["sampleOutput"] = sql_ex_out

                    problem_stmt = q.get("problemStatement") or q.get("scenario") or f"Write a solution for the given {q.get('topic', 'Logic')} task."
                    if any(banned in problem_stmt.lower() for banned in ["solve this coding problem", "complete the program", "write python code."]):
                        problem_stmt = f"Given input for {q.get('topic', 'Logic')}, write a solution to compute and return the correct result."

                    task = q.get("candidateTask") or problem_stmt
                    title_val = q.get("title") or f"{q.get('topic', 'Technical')} Challenge"

                    q["title"] = title_val
                    q["scenario"] = problem_stmt
                    q["problemStatement"] = problem_stmt
                    q["candidateTask"] = task
                    q["question"] = str(q.get("question") or task).strip()

                if q["subject"].upper() == "APTITUDE":
                    q["exampleInput"] = None
                    q["exampleOutput"] = None
                    q["sampleInput"] = None
                    q["sampleOutput"] = None
                else:
                    if not q.get("exampleInput") or is_placeholder(str(q.get("exampleInput")), banned_inputs):
                        q["exampleInput"] = q.get("sampleInput") or ""
                    if not q.get("exampleOutput") or is_placeholder(str(q.get("exampleOutput")), banned_outputs):
                        q["exampleOutput"] = q.get("sampleOutput") or ""

                if not q.get("inputFormat") and q["subject"].upper() != "APTITUDE":
                    q["inputFormat"] = "Standard line of input matching problem parameter requirements."
                if not q.get("outputFormat") and q["subject"].upper() != "APTITUDE":
                    q["outputFormat"] = "Single line containing computed result."
                if not q.get("constraints") and q["subject"].upper() != "APTITUDE":
                    q["constraints"] = ["1 <= N <= 1000, standard execution time and memory limits."]
                elif isinstance(q.get("constraints"), str):
                    q["constraints"] = [q["constraints"]]

                expected = q.get("expectedAnswer") or q.get("correctAnswer") or "Implementation matching requirements."
                q["expectedAnswer"] = expected
                q["correctAnswer"] = expected
                q["solution"] = {"python": expected}

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
2. Review the candidate's answer carefully based on correctness and behavioral output. Point out any syntax errors, logical errors, edge case failures, or performance issues.
3. Accept any valid implementation that produces the correct output (different algorithms, variable names, loops, recursion, list comprehensions, helper functions). Do NOT perform text-based comparison or penalize candidate simply because their code differs from the reference answer.
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
        Evaluates a candidate's answer to a technical assessment question based on correctness,
        functionality, edge case handling, and efficiency, rather than similarity to the reference answer.
        """
        if not candidate_answer or not str(candidate_answer).strip():
            return {
                "score": 0,
                "similarity_score": 0,
                "status": "Not Attempted",
                "ai_explanation": "Question not attempted. No response submitted.",
                "feedback": "Question not attempted. No response submitted.",
                "strengths": "None",
                "missing_points": "No answer submitted.",
                "suggested_improvement": "Provide a complete response to the problem.",
                "improvements": "Provide a complete response to the problem."
            }

        if "def " in str(candidate_answer) or "pass" in str(candidate_answer) or "return" in str(candidate_answer):
            from app.utils.code_evaluator import is_code_attempted
            if not is_code_attempted(candidate_answer):
                return {
                    "score": 0,
                    "similarity_score": 0,
                    "status": "Not Attempted",
                    "ai_explanation": "Question not attempted. Unchanged starter code or non-functional placeholder code submitted.",
                    "feedback": "Question not attempted. Unchanged starter code or non-functional placeholder code submitted.",
                    "strengths": "None",
                    "missing_points": "No solution logic implemented.",
                    "suggested_improvement": "Implement the requested algorithm logic inside the function body.",
                    "improvements": "Implement the requested algorithm logic inside the function body."
                }

        prompt = f"""You are an expert AI technical assessment evaluator. Your task is to evaluate a candidate's Python / technical solution based strictly on correctness, functionality, and execution output, NOT text similarity to the reference answer.

Assessment Context / Question: {question}
Scenario Context: {scenario if scenario else "N/A"}
Question Type: {question_type}

Reference Solution (Benchmark for Expected Behavior ONLY - Do NOT perform text comparison):
{correct_answer}

Candidate's Solution (Code / Response Being Evaluated):
{candidate_answer}

EVALUATION MANDATE & RULES (STRICT COMPLIANCE REQUIRED):
1. EVALUATE ON CORRECTNESS & BEHAVIOR (NOT SIMILARITY):
   - Do NOT perform a line-by-line or text-based comparison with the reference solution.
   - Use the reference solution ONLY as a benchmark for expected output and behavioral requirements.
   - Accept ANY valid programming implementation that produces the correct output or solves the business scenario, even if it uses a completely different algorithm, different variable names, loops instead of built-in functions, recursion instead of iteration, list comprehensions, helper functions, or alternative libraries.
   - NEVER penalize a candidate simply because their code structure or approach differs from the reference solution. Multiple valid approaches must receive the same score if they are correct and efficient.

2. CORE EVALUATION CRITERIA:
   - Correctness: Does the solution logically satisfy the problem requirements, handle edge cases, and avoid syntax/runtime errors?
   - Edge Case Handling & Robustness: Does the solution account for edge cases, null/empty inputs, boundary conditions, or overflow?
   - Algorithmic Efficiency: Is the time and space complexity suitable for the problem constraints?
   - Code Quality & Readability: Is the code clean, well-structured, and readable?

3. SCORING METHODOLOGY (0–100 SCALE):
   - Assign an overall score (0–100) in "score" and "similarity_score" based primarily on correctness, test case logic, and efficiency.
   - Set "status" to:
     * "Correct" if score >= 80
     * "Partially Correct" if score is between 40 and 79
     * "Incorrect" if score < 40

4. STRUCTURED FEEDBACK REQUIREMENTS:
   - "ai_explanation": Comprehensive analysis covering correctness, efficiency, and code quality.
   - "strengths": Key technical strengths and positive aspects of the candidate's implementation.
   - "missing_points": Identified bugs, missing edge-case handling, runtime errors, or logical gaps.
   - "suggested_improvement": Specific, actionable suggestions for performance optimization, readability, or handling edge cases.

CRITICAL RULES:
1. Return ONLY a raw valid JSON object. Do NOT wrap in markdown code blocks (no ```json or ```).
2. The JSON structure must match the schema below.

Response Schema:
{{
  "similarity_score": 90,
  "score": 90,
  "status": "Correct",
  "ai_explanation": "Solution correctly implements the problem requirements using an efficient iterative approach. It handles boundary conditions well and produces accurate results.",
  "strengths": "Clean logic, accurate algorithm implementation, and proper variable naming.",
  "missing_points": "Minor omission in handling empty input list edge cases.",
  "suggested_improvement": "Add a guard clause at the start to handle empty input lists gracefully."
}}"""
        system_message = (
            "You are an expert AI technical assessment evaluator. You evaluate candidate technical solutions "
            "strictly based on correctness, edge-case handling, code quality, and algorithmic efficiency, "
            "NEVER penalizing candidates for using different programming approaches from the reference solution. "
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

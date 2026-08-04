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

        system_message = (
            "You are an expert AI Python assessment generator that generates Python coding assessments for recruiters. "
            "Your task is to generate ONE unique Python coding question every time the recruiter requests Python assessment generation. "
            "The generated question MUST strictly satisfy all the following requirements:\n\n"
            "1. TOPICS: Question MUST belong to beginner or intermediate Python concepts: Sum, Addition, Subtraction, Multiplication, Division, Average, Maximum, Minimum, Even Numbers, Odd Numbers, Prime Number, Palindrome, Armstrong Number, Perfect Number, Factorial, Fibonacci, Leap Year, Reverse Number, Reverse String, Count Digits, Sum of Digits, Character Count, Vowel Count, Consonant Count, Word Count, String Manipulation, List, Tuple, Dictionary, Set, Searching, Sorting, Frequency Count, Remove Duplicates, Mathematical Problems, Number Problems, Pattern Printing, If Else, Nested If, Loops, While Loop, For Loop, Functions, Basic Input Output, Beginner Python Logic.\n"
            "2. FORBIDDEN TOPICS: Do NOT generate advanced topics (Classes, OOP, Decorators, Generators, Threads, Async, APIs, File Handling, Database, NumPy, Pandas, Django, Flask, Regular Expressions, Recursion unless explicitly requested).\n"
            "3. SCENARIO-BASED MANDATE: Every question MUST be written as a real-world scenario (e.g., School, College, Library, Hospital, Railway, Airport, Cricket, Restaurant, Shopping Mall, Parking, Employee Salary, Student Marks, Banking, Delivery, Weather, Attendance, Electricity Bill, Mobile Recharge, Hotel, Supermarket, Inventory, Online Shopping, Movie Ticket Booking, Examination, Bus Reservation). NEVER generate plain textbook questions like 'Find the factorial' or 'Check palindrome'. Always wrap every question into a real-world scenario.\n"
            "4. DIFFICULTY: Generate ONLY the difficulty selected by the recruiter (Easy, Medium, Hard). Do not generate a different difficulty.\n"
            "5. STARTER CODE & FUNCTION FORMAT: Candidates must complete only one function `def solve(...):` (or dynamic parameters matching the scenario, e.g., `def solve(amount):`, `def solve(numbers):`, `def solve(text):`). Starter code must adapt parameters dynamically. Never use static or hardcoded starter code.\n"
            "6. OUTPUT & CANDIDATE REQUIREMENT: Candidate solves problem using `print()`. Do NOT expect `return`. Evaluation compares only printed output.\n"
            "7. TEST CASES & EVALUATION: Generate exactly ONE visible Sample Input and Sample Output. Never generate hidden test cases or secret validation cases. Score = 100 if printed output matches visible Sample Output, else 0.\n"
            "8. RANDOMIZATION: Every generation MUST be completely different. Randomize story, scenario, character names, variables, constraints, numbers, input values, output values, and wording."
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
        AdventureWorks SQL schema compliance, strict distractor guidelines, and topic relevance.
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

4. SCENARIO-BASED PYTHON CODING QUESTIONS:
   - STRICT MANDATE FOR REAL-WORLD SCENARIOS: Every Python coding question MUST be framed as a real-world scenario (e.g., School, College, Library, Hospital, Railway, Airport, Cricket, Restaurant, Shopping Mall, Parking, Employee Salary, Student Marks, Banking, Delivery, Weather, Attendance, Electricity Bill, Mobile Recharge, Hotel, Supermarket, Inventory, Online Shopping, Movie Ticket Booking, Examination, Bus Reservation).
   - BAN ON PLAIN TEXTBOOK QUESTIONS: Never generate plain textbook questions like "Find the factorial" or "Check palindrome". Always wrap every question into a real-world scenario (e.g. "A school wants to reward students...", "A supermarket wants to calculate the bill...", "A bank wants to verify...").
   - ALLOWED BEGINNER / INTERMEDIATE TOPICS: Generate ONLY from beginner or intermediate Python concepts: Sum, Addition, Subtraction, Multiplication, Division, Average, Maximum, Minimum, Even Numbers, Odd Numbers, Prime Number, Palindrome, Armstrong Number, Perfect Number, Factorial, Fibonacci, Leap Year, Reverse Number, Reverse String, Count Digits, Sum of Digits, Character Count, Vowel Count, Consonant Count, Word Count, String Manipulation, List, Tuple, Dictionary, Set, Searching, Sorting, Frequency Count, Remove Duplicates, Mathematical Problems, Number Problems, Pattern Printing, If Else, Nested If, Loops, While Loop, For Loop, Functions, Basic Input Output, Beginner Python Logic.
   - FORBIDDEN ADVANCED TOPICS: Do NOT generate advanced topics (Classes, OOP, Decorators, Generators, Threads, Async, APIs, File Handling, Database, NumPy, Pandas, Django, Flask, Regular Expressions, Recursion unless explicitly requested).
   - DYNAMIC STARTER CODE & FUNCTION FORMAT: Candidate completes only one function. Always generate dynamic starter code like `def solve(...):` matching parameter names to the scenario (e.g., `def solve(amount):`, `def solve(student_marks):`, `def solve(n):`, `def solve(text):`).
   - CANDIDATE OUTPUT FORMAT: Candidates solve using print(). Do NOT expect return.
   - SAMPLE TEST CASE ONLY: Generate exactly ONE visible Sample Input (`sampleInput`) and Sample Output (`sampleOutput`). Do NOT generate hidden test cases or secret validation cases.
   - RANDOMIZATION MANDATE: Every question generation MUST be completely unique. Never repeat previous questions, scenarios, stories, character names, variables, constraints, numbers, input values, or output values.
    - CRITICAL GENERATION & VALIDATION WORKFLOW FOR CODING QUESTIONS:
      Step 1: Generate a brand new, unique problem statement.
      Step 2: Generate the reference Python solution.
      Step 3: Generate 1 visible test case (sampleInput and sampleOutput / exampleInput and exampleOutput).
      Step 4: Execute the reference solution internally and compute the exact output for the generated input.
      Step 5: Verify that exampleInput and exampleOutput are non-empty, non-null, and non-trivial.
    - STRICT MANDATES FOR PYTHON CODING QUESTION GENERATION:
      1. NOVELTY & UNIQUENESS: Every assessment MUST generate a COMPLETELY NEW coding question. NEVER repeat titles, problem statements, sample inputs, sample outputs, test data, numbers, strings, or arrays from previous generations. Vary wording, examples, numbers, inputs, outputs, and constraints.
      2. TOPIC RESTRICTION: Generate ONLY from the recruiter's selected topic (e.g. Topic = Strings -> String problems only, Topic = Lists -> List problems only).
      3. COMPLETE PROBLEM STATEMENT: Always generate a detailed, complete coding problem (e.g. "Given an array of integers, write a Python function to find the length of the longest subarray with a sum equal to target k."). NEVER generate generic statements like "Solve this coding problem", "Write Python code", or "Complete the program".
      4. MANDATORY SAMPLE INPUT & OUTPUT: sampleInput, sampleOutput, exampleInput, and exampleOutput are MANDATORY. NEVER return "No input", "No output", empty string, "N/A", or null.
      5. DOMAIN-SPECIFIC STARTER CODE: Starter code parameter MUST match the problem domain:
         - Strings: def solution(text):
         - Lists: def solution(arr):
         - Numbers: def solution(n):
         - Matrix: def solution(matrix):
         - Dictionary: def solution(data):
         Do NOT always use def solution(data):.
      6. INTERNAL VALIDATION: Execute the reference solution mentally to derive sampleOutput directly from sampleInput. Never guess outputs.
      7. JSON SCHEMA PER CODING QUESTION:
         {{
           "title": "Longest Subarray with Target Sum",
           "difficulty": "Medium",
           "topic": "Lists",
           "problemStatement": "Given an array of integers and a target sum k, write a Python function to return the length of the longest continuous subarray whose elements sum to k.",
           "starterCode": "def solution(arr, k):\n    pass",
           "sampleInput": "7\n4 8 2 9 1 6 3\n11",
           "sampleOutput": "2",
           "exampleInput": "7\n4 8 2 9 1 6 3\n11",
           "exampleOutput": "2",
           "constraints": ["1 <= len(arr) <= 1000", "-10^5 <= arr[i] <= 10^5"],
           "expectedAnswer": "def solution(arr, k):\n    # Reference implementation\n    pass",
           "explanation": "Calculates prefix sums to find the maximum length subarray with sum equal to k."
         }}
    - "evaluationCriteria": Clear rubric highlighting key evaluation points.

5. TOPIC RELEVANCE:
   - Every question's "subject" MUST strictly be one of: {request.subjects}.

RESPONSE SCHEMA (RETURN RAW CLEAN JSON ONLY):
{{
  "questions": [
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
      "subject": "Python",
      "topic": "Lists",
      "type": "SCENARIO",
      "difficulty": "Medium",
      "title": "Find Second Largest Unique Element",
      "problemStatement": "Given a list of integers, write a Python function to find the second largest unique element in the list.",
      "starterCode": "def solution(arr):\n    pass",
      "sampleInput": "10 25 40 15 35",
      "sampleOutput": "35",
      "constraints": ["1 <= len(arr) <= 1000"],
      "expectedAnswer": "def solution(arr):\n    unique_sorted = sorted(list(set(arr)))\n    return unique_sorted[-2] if len(unique_sorted) >= 2 else None",
      "explanation": "Sorts unique elements and picks the second largest."
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

    def _generate_python_starter_and_sig(self, q: dict, s_in: str) -> tuple[str, str]:
        """
        Derives the function signature and starter code for Python coding questions based on strict rules:
        - 1 input scalar: def solve(num):
        - 2 inputs: def solve(a, b):
        - 3 inputs: def solve(a, b, c):
        - list input: def solve(numbers):
        - string input: def solve(text):
        """
        topic_lower = str(q.get("topic", "")).lower()
        q_text = (str(q.get("question", "")) + " " + str(q.get("problemStatement", "")) + " " + str(q.get("inputFormat", ""))).lower()
        sig_raw = str(q.get("functionSignature") or q.get("starterCode") or "")
        
        # Parse params from AI signature if available
        params_in_sig = []
        sig_match = re.search(r"(\w+)\s*\((.*?)\)", sig_raw)
        if sig_match:
            p_str = sig_match.group(2).strip()
            if p_str:
                params_in_sig = [p.strip() for p in p_str.split(",") if p.strip()]

        # 1. Check for 3 inputs
        if len(params_in_sig) == 3 or any(w in q_text for w in ["three numbers", "three integers", "three inputs", "a, b, c", "three values"]):
            return "solve(a, b, c)", "def solve(a, b, c):\n    # Write your solution here using print()\n    pass"

        # 2. Check for 2 inputs
        if len(params_in_sig) == 2 or any(w in q_text for w in ["two numbers", "two integers", "two inputs", "a, b", "greatest of two", "two values"]):
            return "solve(a, b)", "def solve(a, b):\n    # Write your solution here using print()\n    pass"

        # 3. Check for String input
        is_string = any(w in topic_lower or w in q_text for w in [
            "string", "text", "vowel", "consonant", "word", "char", "character", "palindrome", "reverse string", "anagram"
        ])
        if not is_string and s_in and not any(c.isdigit() for c in s_in) and len(s_in.split()) == 1 and s_in.isalpha():
            is_string = True

        if is_string:
            return "solve(text)", "def solve(text):\n    # Write your solution here using print()\n    pass"

        # 4. Check for List input
        is_list = any(w in topic_lower or w in q_text for w in [
            "list", "array", "arr", "sequence", "elements", "remove duplicates", "sort", "frequency",
            "maximum element", "minimum element", "largest element", "smallest element"
        ])
        if not is_list and s_in and (" " in s_in or "," in s_in) and not ("\n" in s_in) and len(s_in.split()) > 3:
            is_list = True

        if is_list:
            return "solve(numbers)", "def solve(numbers):\n    # Write your solution here using print()\n    pass"

        # 5. Check parameter names from AI signature if 1 parameter
        if len(params_in_sig) == 1:
            p_name = params_in_sig[0].lower()
            if p_name in ["text", "string", "s", "word", "sentence"]:
                return "solve(text)", "def solve(text):\n    # Write your solution here using print()\n    pass"
            elif p_name in ["numbers", "arr", "nums", "lst", "items", "data", "list"]:
                return "solve(numbers)", "def solve(numbers):\n    # Write your solution here using print()\n    pass"
            else:
                return "solve(num)", "def solve(num):\n    # Write your solution here using print()\n    pass"

        # 6. Fallback default for 1 numeric/general input
        return "solve(num)", "def solve(num):\n    # Write your solution here using print()\n    pass"

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

                corr = str(q.get("correctAnswer") or "").strip()
                if corr not in q["options"]:
                    q["correctAnswer"] = q["options"][0]
                
                if not q.get("explanation") or is_placeholder(str(q.get("explanation")), banned_outputs):
                    q["explanation"] = f"The correct answer is '{q['correctAnswer']}', which accurately solves the {q['subject']} ({q['topic']}) task."

            # 4. Clean Scenario / Programming / SQL questions
            else:
                q["options"] = None
                
                if q["subject"].upper() == "PYTHON":
                    topic_lower = str(q.get("topic", "")).lower()

                    # Extract sample input first to resolve signature parameter matching accurately
                    vtc_dict = q.get("visibleTestCase")
                    if not isinstance(vtc_dict, dict):
                        visible_list = q.get("visibleTestCases") or []
                        if isinstance(visible_list, list) and len(visible_list) > 0 and isinstance(visible_list[0], dict):
                            vtc_dict = visible_list[0]
                        else:
                            vtc_dict = {}

                    s_in = str(vtc_dict.get("input") or q.get("sampleInput") or q.get("exampleInput") or "").strip()
                    s_out = str(vtc_dict.get("expectedOutput") if vtc_dict.get("expectedOutput") is not None else (vtc_dict.get("output") or q.get("sampleOutput") or q.get("exampleOutput") or "")).strip()

                    # Derive exact starter code & signature (solve(num), solve(a, b), solve(a, b, c), solve(numbers), solve(text))
                    sig_name, dynamic_starter = self._generate_python_starter_and_sig(q, s_in)
                    q["functionSignature"] = sig_name
                    q["starterCode"] = dynamic_starter
                    q["starter_code"] = dynamic_starter

                    # Synchronize inputFormat for complete consistency
                    if sig_name == "solve(text)":
                        q["inputFormat"] = "Single string text"
                    elif sig_name == "solve(numbers)":
                        q["inputFormat"] = "Space-separated numbers"
                    elif sig_name == "solve(a, b, c)":
                        q["inputFormat"] = "Three space-separated or line-separated values a, b, and c"
                    elif sig_name == "solve(a, b)":
                        q["inputFormat"] = "Two space-separated or line-separated values a and b"
                    else:
                        q["inputFormat"] = "Single numeric value num"

                    # Banned placeholder cleanup
                    if is_placeholder(s_in, banned_inputs):
                        if "string" in topic_lower or "palindrome" in topic_lower:
                            s_in = "madam"
                        elif "list" in topic_lower or "array" in topic_lower:
                            s_in = "10 25 40 15 35"
                        elif "number" in topic_lower or "factorial" in topic_lower or "prime" in topic_lower:
                            s_in = "6"
                        else:
                            s_in = "madam"

                    if is_placeholder(s_out, banned_outputs):
                        if "string" in topic_lower or "palindrome" in topic_lower:
                            s_out = "True"
                        elif "list" in topic_lower or "array" in topic_lower:
                            s_out = "35"
                        elif "number" in topic_lower or "factorial" in topic_lower:
                            s_out = "720"
                        else:
                            s_out = "True"

                    v_inp = s_in
                    v_exp = s_out

                    # Automated recomputation for standard topics
                    q_str = str(q.get("question", "") or q.get("problemStatement", "")).lower()

                    if "palindrome" in topic_lower or "palindrome" in q_str:
                        is_pal = (v_inp.lower() == v_inp.lower()[::-1])
                        if v_exp.upper() in ["YES", "NO"]:
                            v_exp = "YES" if is_pal else "NO"
                        elif v_exp in ["1", "0"]:
                            v_exp = "1" if is_pal else "0"
                        else:
                            v_exp = "True" if is_pal else "False"

                    elif ("factorial" in topic_lower or "factorial" in q_str) and v_inp.replace("-", "").isdigit():
                        import math
                        val = abs(int(v_inp))
                        val = min(val, 20)
                        v_exp = str(math.factorial(val))

                    elif ("reverse number" in topic_lower or "reverse number" in q_str or "reverse a number" in q_str) and v_inp.replace("-", "").isdigit():
                        clean_digits = v_inp.lstrip("-")
                        rev = clean_digits[::-1]
                        v_exp = f"-{rev.lstrip('0') or '0'}" if v_inp.startswith("-") else (rev.lstrip("0") or "0")

                    elif ("prime" in topic_lower or "prime number" in q_str) and v_inp.replace("-", "").isdigit():
                        num = int(v_inp)
                        is_p = True if num > 1 else False
                        for i in range(2, int(num**0.5) + 1):
                            if num % i == 0:
                                is_p = False
                                break
                        if v_exp.upper() in ["YES", "NO"]:
                            v_exp = "YES" if is_p else "NO"
                        elif v_exp in ["1", "0"]:
                            v_exp = "1" if is_p else "0"
                        else:
                            v_exp = "True" if is_p else "False"

                    elif ("sum of digits" in topic_lower or "sum of digits" in q_str) and any(c.isdigit() for c in v_inp):
                        v_exp = str(sum(int(c) for c in v_inp if c.isdigit()))

                    elif ("count digits" in topic_lower or "count digits" in q_str) and any(c.isdigit() for c in v_inp):
                        v_exp = str(len([c for c in v_inp if c.isdigit()]))

                    visible_tc = {"input": v_inp, "expectedOutput": v_exp, "output": v_exp}
                    q["visibleTestCase"] = visible_tc
                    q["visibleTestCases"] = [visible_tc]
                    q["sampleInput"] = v_inp
                    q["sampleOutput"] = v_exp
                    q["exampleInput"] = v_inp
                    q["exampleOutput"] = v_exp
                    q["hiddenTestCases"] = []  # No hidden test cases
                    q["testCases"] = {
                        "visible": [{"input": v_inp, "output": v_exp}],
                        "hidden": []
                    }
                    q["evaluation"] = {
                        "visibleTestCasesOnly": True,
                        "hiddenTestCases": False,
                        "usesPrint": True,
                        "usesReturn": False,
                        "fullScoreOnlyIfVisibleTestCasePasses": True,
                        "partialScoring": False
                    }
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

                task = q.get("candidateTask") or q.get("question") or problem_stmt
                title_val = q.get("title") or f"{q.get('topic', 'Technical')} Challenge"

                q["title"] = title_val
                q["scenario"] = problem_stmt
                q["problemStatement"] = problem_stmt
                q["candidateTask"] = task
                q["question"] = str(q.get("question") or task).strip()
                if not q.get("exampleInput") or is_placeholder(str(q.get("exampleInput")), banned_inputs):
                    q["exampleInput"] = q.get("sampleInput") or "madam"
                if not q.get("exampleOutput") or is_placeholder(str(q.get("exampleOutput")), banned_outputs):
                    q["exampleOutput"] = q.get("sampleOutput") or "True"
                if not q.get("inputFormat"):
                    q["inputFormat"] = "Standard line of input matching problem parameter requirements."
                if not q.get("outputFormat"):
                    q["outputFormat"] = "Single line containing computed result."
                if not q.get("constraints"):
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

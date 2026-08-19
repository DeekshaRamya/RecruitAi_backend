import json
import logging
import re
from typing import Dict, Any, List, Optional
from app.core.azure_openai import AzureOpenAIClient
from app.schemas.assessment import AssessmentGenerateRequest
from app.schemas.interview import InterviewGenerateRequest, InterviewEvaluateRequest
from app.services.ai_usage_service import AiFeature

logger = logging.getLogger("recruitai-backend.azure_openai_service")

def validate_sql_question(q: dict) -> bool:
    """
    Validates SQL Scenario-Based questions against strict standardized template rules:
    ✓ Scenario exists (2–4 sentences, business context)
    ✓ Task exists (explicit instructions)
    ✓ Input/Output Format exists ("Query the live database and return these columns...")
    ✓ Every required table is explicitly mentioned (schema.tablename)
    ✓ Every required table explains why it is being used (e.g., "(to get ...)")
    ✓ Required columns are explicitly mentioned
    ✓ JOIN conditions explicitly mentioned if query uses JOIN
    ✓ WHERE filtering conditions explicitly mentioned if query uses WHERE
    ✓ GROUP BY specified if query uses GROUP BY
    ✓ HAVING specified if query uses HAVING
    ✓ ORDER BY specified if query uses ORDER BY
    ✓ Output columns are listed in order
    """
    scenario = str(q.get("scenario") or "").strip()
    task = str(q.get("task") or q.get("candidateTask") or "").strip()
    prob_stmt = str(q.get("problemStatement") or q.get("question") or "").strip()
    io_fmt = str(q.get("inputOutputFormat") or q.get("inputFormat") or "").strip()
    expected_query = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()

    combined_text = f"{scenario} {task} {prob_stmt} {io_fmt}".lower()
    query_lower = expected_query.lower()

    if not scenario or len(scenario) < 15:
        logger.warning(f"[SQL Validation Fail] Missing or short scenario: '{scenario}'")
        return False

    if not task and "task:" not in prob_stmt.lower():
        logger.warning("[SQL Validation Fail] Missing task specification.")
        return False

    if not io_fmt and "input/output format" not in prob_stmt.lower() and "return these columns" not in combined_text:
        logger.warning("[SQL Validation Fail] Missing Input/Output format specification.")
        return False

    if not any(schema in combined_text for schema in ["humanresources.", "sales.", "production.", "purchasing.", "person."]):
        logger.warning("[SQL Validation Fail] Database schema table names not explicitly mentioned.")
        return False

    if not any(why_kw in combined_text for why_kw in ["(to get", "to get", "to retrieve", "using", "for"]):
        logger.warning("[SQL Validation Fail] Explanation of why tables are used is missing.")
        return False

    if not any(kw in combined_text for kw in ["column", "select", "retrieve", "return", "columns"]):
        logger.warning("[SQL Validation Fail] Required columns not explicitly mentioned.")
        return False

    if "join" in query_lower and not any(w in combined_text for w in ["join", "on", "combine", "matching", "related", "connected", "table", "using"]):
        logger.warning("[SQL Validation Fail] Query uses JOIN but JOIN condition is not specified in Task.")
        return False

    if "where" in query_lower and not any(w in combined_text for w in ["where", "filter", "active", "flag", "equal", "only", "with", "for", "in", "greater", "less", "whose", "status"]):
        logger.warning("[SQL Validation Fail] Query uses WHERE but filtering condition is not specified in Task.")
        return False

    if "group by" in query_lower and not any(w in combined_text for w in ["group", "aggregate", "total", "by", "per", "each", "count", "sum", "avg", "summary"]):
        logger.warning("[SQL Validation Fail] Query uses GROUP BY but GROUP BY requirement is not specified in Task.")
        return False

    if "having" in query_lower and not any(w in combined_text for w in ["having", "greater", "more", "filter", "exceed", "limit", "than"]):
        logger.warning("[SQL Validation Fail] Query uses HAVING but HAVING requirement is not specified in Task.")
        return False

    if "order by" in query_lower and not any(w in combined_text for w in ["order", "sort", "top", "highest", "lowest", "rank", "list", "descending", "ascending", "alphabetical"]):
        logger.warning("[SQL Validation Fail] Query uses ORDER BY but ORDER BY requirement is not specified in Task.")
        return False

    return True


def validate_python_question(q: dict) -> bool:
    """
    Validates Python Scenario-Based questions against the strict template rules:
    - Scenario exists
    - Task exists
    - Input/Output Format exists
    - Example Input exists
    - Example Output exists
    - Function name explicitly mentioned (e.g. solution)
    - Parameter names explicitly mentioned
    - Input type specified
    - Output type specified
    - Example is logically correct (valid sample input/output strings)
    - Starter code matches function name and parameters in the Task
    """
    scenario = str(q.get("scenario") or "").strip()
    task = str(q.get("task") or q.get("candidateTask") or "").strip()
    prob_stmt = str(q.get("problemStatement") or q.get("question") or "").strip()
    in_fmt = str(q.get("inputFormat") or "").strip()
    out_fmt = str(q.get("outputFormat") or "").strip()
    sample_in = str(q.get("sampleInput") or q.get("exampleInput") or "").strip()
    sample_out = str(q.get("sampleOutput") or q.get("exampleOutput") or "").strip()
    starter = str(q.get("starterCode") or q.get("starter_code") or "").strip()

    combined_text = f"{scenario} {task} {prob_stmt}".lower()

    is_python = str(q.get("subject") or "").strip().upper() == "PYTHON"

    if not scenario or len(scenario) < 15:
        logger.warning(f"[Python Validation Fail] Missing scenario: '{scenario}'")
        return False

    if not task and "task:" not in prob_stmt.lower():
        logger.warning("[Python Validation Fail] Missing task specification.")
        return False

    if not in_fmt and not is_python and "input type:" not in combined_text and "input format:" not in combined_text and "input is a" not in combined_text:
        logger.warning("[Python Validation Fail] Missing input format specification.")
        return False

    if not out_fmt and not is_python and "output type:" not in combined_text and "output format:" not in combined_text and "output is a" not in combined_text and "return type:" not in combined_text:
        logger.warning("[Python Validation Fail] Missing output format specification.")
        return False

    if not sample_in or any(b in sample_in.lower() for b in ["placeholder", "tbd", "n/a", "no input"]):
        logger.warning(f"[Python Validation Fail] Missing or invalid example input: '{sample_in}'")
        return False

    if not sample_out or any(b in sample_out.lower() for b in ["placeholder", "tbd", "n/a", "no output"]):
        logger.warning(f"[Python Validation Fail] Missing or invalid example output: '{sample_out}'")
        return False

    sig_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", starter)
    if not sig_match:
        sig_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", starter)

    if not sig_match:
        logger.warning("[Python Validation Fail] Cannot extract function signature from starter code.")
        return False

    func_name_in_starter = sig_match.group(1)
    if func_name_in_starter.lower() not in combined_text.lower() and func_name_in_starter not in starter:
        logger.warning(f"[Python Validation Fail] Function name '{func_name_in_starter}' not mentioned in Task.")
        return False

    params = [p.strip() for p in sig_match.group(2).split(",") if p.strip()]
    if not params:
        logger.warning("[Python Validation Fail] Parameter names not specified in function signature.")
        return False

    for param in params:
        if param not in combined_text and param not in starter:
            logger.warning(f"[Python Validation Fail] Parameter '{param}' not mentioned in Task text.")
            return False

    if not is_python and not any(kw in f"{in_fmt} {combined_text}".lower() for kw in ["list", "int", "str", "float", "matrix", "boolean", "array", "text", "number"]):
        logger.warning("[Python Validation Fail] Input type not explicitly specified.")
        return False

    if not is_python and not any(kw in f"{out_fmt} {combined_text}".lower() for kw in ["list", "int", "str", "float", "matrix", "boolean", "return", "integer", "string"]):
        logger.warning("[Python Validation Fail] Output type not explicitly specified.")
        return False

    return True


def is_duplicate_or_similar(
    q: dict, 
    existing_questions: Optional[List[Any]] = None, 
    generated_so_far: Optional[List[dict]] = None
) -> bool:
    """
    Compares a generated question with previously generated questions (across assessments
    and within the current generation session).
    Returns True if similarity is high (same scenario, same business context, same logic, 
    or same solution approach), meaning it must be discarded.
    """
    from difflib import SequenceMatcher

    def get_clean_text(item: Any) -> str:
        if isinstance(item, dict):
            s = f"{item.get('topic', '')} {item.get('scenario', '')} {item.get('task', '')} {item.get('problemStatement', '')} {item.get('question', '')} {item.get('expectedAnswer', '')}"
        else:
            s = str(item or "")
        s = re.sub(r'[^\w\s]', ' ', s.lower())
        return " ".join(s.split())

    q_text = get_clean_text(q)
    if not q_text or len(q_text) < 10:
        return False

    q_words = set(q_text.split())

    # 1. Compare against existing questions across assessments
    if existing_questions:
        for ex in existing_questions:
            ex_text = get_clean_text(ex)
            if not ex_text or len(ex_text) < 10:
                continue

            seq_ratio = SequenceMatcher(None, q_text, ex_text).ratio()
            if seq_ratio > 0.60:
                logger.warning(f"[Similarity Filter] Question discarded: high sequence similarity ({seq_ratio:.2f}) with existing question.")
                return True

            ex_words = set(ex_text.split())
            if q_words and ex_words:
                intersection = len(q_words.intersection(ex_words))
                union = float(len(q_words.union(ex_words)))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > 0.58:
                    logger.warning(f"[Similarity Filter] Question discarded: high word Jaccard similarity ({jaccard:.2f}) with existing question.")
                    return True

    # 2. Compare against questions generated so far in current assessment session
    if generated_so_far:
        for prev in generated_so_far:
            prev_text = get_clean_text(prev)
            if not prev_text or len(prev_text) < 10:
                continue

            seq_ratio = SequenceMatcher(None, q_text, prev_text).ratio()
            if seq_ratio > 0.60:
                logger.warning(f"[Similarity Filter] Question discarded: high sequence similarity ({seq_ratio:.2f}) with session question.")
                return True

            prev_words = set(prev_text.split())
            if q_words and prev_words:
                intersection = len(q_words.intersection(prev_words))
                union = float(len(q_words.union(prev_words)))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > 0.58:
                    logger.warning(f"[Similarity Filter] Question discarded: high word Jaccard similarity ({jaccard:.2f}) with session question.")
                    return True

            q_topic = str(q.get("topic", "")).strip().lower()
            prev_topic = str(prev.get("topic", "")).strip().lower()
            q_ans = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip().lower()
            prev_ans = str(prev.get("expectedAnswer") or prev.get("correctAnswer") or "").strip().lower()
            if q_topic and q_topic == prev_topic and q_ans and q_ans == prev_ans:
                logger.warning(f"[Similarity Filter] Question discarded: identical topic and expected answer with session question.")
                return True

    return False

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAIClient()

    async def generate_questions(
        self, 
        request: AssessmentGenerateRequest,
        existing_questions: Optional[List[str]] = None,
        exclude_sql_scenarios: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Builds the prompt with a dynamic uniqueness seed and existing assessment exclusions,
        calls Azure OpenAI API client, cleans the response, and parses & validates it into high-quality,
        non-repetitive question objects.
        """
        # 1. Dynamically construct prompt
        prompt = self._build_prompt(request, existing_questions=existing_questions, exclude_sql_scenarios=exclude_sql_scenarios)
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
            "- Scenario-Based questions: Strictly NO options (`options`: null). Descriptive, calculation-oriented aptitude problems where candidates type the answer in a single input field (`placeholder`: 'Enter your answer' or 'Type your answer here', `answerType`: 'NUMBER' or 'TEXT'). First generate `expectedAnswer`, then analyze `expectedAnswer` and set `outputFormat` strictly to ONLY the concise data type name (`Integer`, `Decimal`, `Percentage`, `Fraction`, `Ratio`, `Time`, `Currency`, `Boolean`, or `String`). Do NOT include any additional instructions, rules, explanations, or bullet points.\n"
            "- Difficulty calibration: Easy (formula-based), Medium (2+ calculation steps), Hard (complex multi-step reasoning).\n\n"
            "PYTHON & TECHNICAL ASSESSMENT MANDATE:\n"
            "- Python coding questions must be framed as real-world scenarios with beginner/intermediate concepts, dynamic starter code, and visible sample test case.\n"
            "- SQL questions must adhere strictly to AdventureWorks schema.\n"
            "- Every generated question MUST be completely unique and non-repetitive."
        )

        # Determine specific feature name from subjects
        feature_name = AiFeature.ASSESSMENT_EVALUATION
        if any("PYTHON" in str(s).upper() for s in request.subjects):
            feature_name = AiFeature.PYTHON_QUESTION_GENERATION
        elif any("APTITUDE" in str(s).upper() for s in request.subjects):
            feature_name = AiFeature.APTITUDE_QUESTION_GENERATION
        elif any("SQL" in str(s).upper() for s in request.subjects):
            feature_name = AiFeature.SQL_QUESTION_GENERATION
        elif any("ENGLISH" in str(s).upper() for s in request.subjects):
            feature_name = AiFeature.ENGLISH_QUESTION_GENERATION

        # 2. Invoke Azure OpenAI via the client
        raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=feature_name)
        
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
        return self._clean_and_validate_questions(data["questions"], request.subjects, existing_questions=existing_questions)

    async def verify_python_code_hardcoding(
        self,
        question_text: str,
        function_signature: str,
        candidate_code: str,
        test_cases: List[Dict[str, Any]],
        ast_reasons: List[str]
    ) -> Dict[str, Any]:
        """
        Stage 3: Invokes Azure OpenAI to verify whether a suspicious Python code submission
        has hardcoded answers to pass visible test cases instead of implementing the required algorithm.
        Returns JSON: {"hardcoded": bool, "confidence": float, "reason": str}
        """
        test_cases_str = ""
        for idx, tc in enumerate(test_cases):
            inp = tc.get("input", "")
            exp_out = tc.get("expectedOutput", "")
            act_out = tc.get("actualOutput", tc.get("actual", ""))
            test_cases_str += (
                f"Test Case #{idx+1}:\n"
                f"  Input: {inp}\n"
                f"  Expected Output: {exp_out}\n"
                f"  Actual Candidate Output: {act_out}\n\n"
            )

        reasons_str = "\n".join([f"- {r}" for r in ast_reasons]) if ast_reasons else "- AST flagged suspicious structure."

        prompt = f"""AUDIT REQUEST: DETERMINE IF CANDIDATE PYTHON CODE IS HARDCODED TO TRICK TEST CASES

QUESTION / PROBLEM STATEMENT:
{question_text}

EXPECTED FUNCTION SIGNATURE:
{function_signature}

CANDIDATE SUBMITTED CODE:
```python
{candidate_code}
```

STATIC AST ANALYSIS FLAGS:
{reasons_str}

TEST CASES & CANDIDATE OUTPUTS:
{test_cases_str}

TASK:
Analyze the candidate's Python code against the problem statement and test cases.
Determine whether the candidate has genuinely implemented the required algorithm/logic, OR if the code is hardcoded (e.g. returning constant literal strings/numbers, hardcoded if-else checks matching visible inputs, or ignoring parameters to bypass test cases).

MANDATORY RESPONSE FORMAT:
Return ONLY a valid JSON object matching this schema (no markdown, no additional commentary):
{{
  "hardcoded": true,
  "confidence": 0.98,
  "reason": "Detailed explanation of why the submission is hardcoded or genuine."
}}"""

        system_msg = (
            "You are an expert AI software engineering assessor and security auditor. "
            "Your job is to detect hardcoded solutions in programming assessments and return clean JSON."
        )

        try:
            raw_res = await self.client.generate_chat_completion(prompt, system_msg, feature_name=AiFeature.PYTHON_ANSWER_EVALUATION)
            cleaned_json = self._clean_json(raw_res)
            data = json.loads(cleaned_json)

            hardcoded = bool(data.get("hardcoded", True))
            confidence = float(data.get("confidence", 0.9))
            reason = str(data.get("reason") or "Code contains hardcoded responses matching test case outputs.").strip()

            logger.info(f"[AI Verification Result] hardcoded={hardcoded}, confidence={confidence}, reason='{reason}'")
            return {
                "hardcoded": hardcoded,
                "confidence": confidence,
                "reason": reason
            }
        except Exception as e:
            logger.error(f"[AzureOpenAIService] AI Verification for hardcoded code failed: {e}. Defaulting to AST findings.")
            return {
                "hardcoded": True,
                "confidence": 0.85,
                "reason": f"AST Analysis flagged suspicious code structure ({'; '.join(ast_reasons[:2])})."
            }

    async def generate_dynamic_sql_scenarios(
        self,
        count: int,
        difficulty_distribution: Any,
        existing_questions: Optional[List[str]] = None,
        generated_so_far: Optional[List[dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Calls Azure OpenAI API client to generate brand-new, unique SQL Scenario questions
        strictly grounded in the live AdventureWorks database schema.
        """
        import uuid
        import time
        from app.services.sql_schema_service import SqlSchemaService

        seed = f"sql-scenario-{uuid.uuid4().hex[:8]}-{int(time.time()*1000)}"
        schema_text = SqlSchemaService.get_live_schema_text()

        easy_pct = getattr(difficulty_distribution, 'easy', 33)
        medium_pct = getattr(difficulty_distribution, 'medium', 34)
        easy_count = round(count * easy_pct / 100.0)
        medium_count = round(count * medium_pct / 100.0)
        hard_count = count - (easy_count + medium_count)
        if hard_count < 0:
            hard_count = 0

        ex_context = ""
        if existing_questions:
            clean_ex = [f"- {q}" for q in existing_questions if q and len(str(q).strip()) > 5]
            if clean_ex:
                ex_context = "\nEXISTING QUESTIONS TO STRICTLY EXCLUDE (DO NOT REUSE OR REPEAT):\n" + "\n".join(clean_ex[:30]) + "\n"

        prompt = f"""Generate {count} BRAND NEW, 100% UNIQUE AdventureWorks Scenario-Based SQL Assessment Questions for candidates.

UNIQUE GENERATION SEED: {seed}
{ex_context}
LIVE ADVENTUREWORKS DATABASE SCHEMA DETAILS:
{schema_text}

TARGET QUESTION COUNT & DIFFICULTY BREAKDOWN:
- Total SQL Scenario Questions Required: {count}
- Difficulty Distribution: Easy ~{easy_count}, Medium ~{medium_count}, Hard ~{hard_count}

MANDATORY RULES FOR EVERY GENERATED SQL QUESTION (STRICT STANDARDIZED TEMPLATE & HIGH READABILITY):

1. EVERY SQL QUESTION MUST FOLLOW EXACTLY THIS 3-PART FORMAT (DO NOT ALTER SECTION NAMES OR ORDER):

### Scenario:
Write a realistic business scenario in 2–4 sentences.

---

### Task:
Write the task using short, readable sentences. NEVER generate a single long paragraph.

Whenever a table is mentioned, write it on its OWN logical line using this exact format:
Use **Schema.TableName** (to get Column1, Column2, Column3).

Example table usage lines:
Use **Production.ProductInventory** (to get ProductID and Quantity).
Use **Production.Product** (to get Name using ProductID).

After explaining each table on its own line, list the remaining requirements separately as clean bullet points:
- Return ProductID, Name and TotalQuantity.
- Filter only active records where SalariedFlag = 1.
- Group the rows by product.
- Sort by TotalQuantity descending.
- Then sort by ProductID ascending.

---

### Input/Output Format:
Query the live database and return these columns:

- Column1
- Column2
- Column3

2. STRICT READABILITY RULES:
   - Bold every database table name (e.g. **Production.ProductInventory**).
   - Immediately explain purpose inside parentheses (to get Column1, Column2).
   - Write every table explanation on its own line.
   - List filtering, joins, grouping, sorting, and output requirements as bullet points.
   - NEVER create one long paragraph for the Task section.

3. SOLUTION QUERY ("expectedAnswer"):
   - Must be a 100% valid T-SQL query executable on SQL Server against the AdventureWorks database.
   - Do NOT include "TOP 5", "LIMIT 5", "FETCH FIRST", or any row-limiting clause in the query UNLESS the task explicitly instructs the candidate to return only a specific limited number of rows (e.g., 'Return top 5 rows'). Write the complete reference query returning the full matching dataset.
   - NEVER use imaginary tables like 'orders', 'users', or 'evaluation_records'. ONLY use tables defined in the schema above.

4. UNIQUENESS & NOVELTY MANDATE:
   - Every question must test a DIFFERENT business scenario and database table across the 5 schemas (HumanResources, Sales, Production, Purchasing, Person).
   - DO NOT repeat or reuse previously generated scenarios, business context, or queries.

RETURN ONLY A CLEAN JSON OBJECT WITH THE FOLLOWING SCHEMA:
{{
  "questions": [
    {{
      "subject": "SQL",
      "topic": "<String: Topic name generated dynamically>",
      "type": "SCENARIO",
      "difficulty": "<String: Easy|Medium|Hard>",
      "scenario": "<String: Dynamically generated 2-4 sentence business scenario>",
      "task": "<String: Dynamically generated task mentioning exact schema tables, why used, columns, joins, filter, group by, order by>",
      "inputOutputFormat": "Input/Output Format: Query the live database and return these columns:\\n\\n<Column1>\\n\\n<Column2>",
      "problemStatement": "### Scenario:\\n<Scenario text>\\n\\n---\\n\\n### Task:\\n<Task text>\\n\\n---\\n\\n### Input/Output Format:\\n<Input/Output Format text>",
      "candidateTask": "<String: Concise task summary>",
      "expectedAnswer": "<String: Dynamically generated standard T-SQL query returning full matching dataset>",
      "evaluationCriteria": "<String: Candidate evaluation criteria>",
      "explanation": "<String: Detailed step-by-step query explanation>"
    }}
  ]
}}"""

        system_msg = (
            "You are an expert SQL assessment developer. You generate brand-new, unique T-SQL scenario questions for recruitment exams strictly based on the AdventureWorks SQL Server database schema."
        )

        raw_response = await self.client.generate_chat_completion(prompt, system_msg, feature_name=AiFeature.SQL_QUESTION_GENERATION)
        cleaned_json = self._clean_json(raw_response)
        try:
            data = json.loads(cleaned_json)
            return data.get("questions", [])
        except Exception as e:
            logger.error(f"[AzureOpenAIService] Failed to parse dynamic SQL scenario response: {e}. Raw response: {raw_response}")
            return []

    def _build_prompt(
        self, 
        request: AssessmentGenerateRequest,
        existing_questions: Optional[List[str]] = None,
        exclude_sql_scenarios: bool = False
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

        sql_exclusion_instruction = ""
        if exclude_sql_scenarios:
            sql_exclusion_instruction = (
                "\nSPECIAL MANDATE FOR SQL QUESTIONS:\n"
                "- DO NOT generate any SQL Scenario-Based questions (type 'SCENARIO' for subject 'SQL'). "
                "All SQL Scenario questions have ALREADY been generated separately.\n"
                "- Generate only MCQ questions for SQL if SQL is included, and MCQs or Scenario questions for non-SQL subjects.\n"
            )

        prompt = f"""Generate professional technical and aptitude recruitment assessment questions based on the following configurations:

UNIQUE ASSESSMENT GENERATION SEED: {unique_generation_seed}
Selected Subjects: {subjects_str}
{sql_schema_context}
{exclusion_context}
{sql_exclusion_instruction}
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
     * MCQ Questions (`"type"`: `"MCQ"`):
       - Formulate a clear placement-exam style question.
       - Provide EXACTLY FOUR realistic option choices in `"options"`.
       - Exactly ONE correct option choice in `"correctAnswer"`.
       - Detailed step-by-step mathematical or logical reasoning in `"explanation"`.
     * Scenario-Based Questions (`"type"`: `"SCENARIO"`):
       - DO NOT generate options (`"options"` MUST BE null / omitted). NEVER generate MCQ options for Scenario-Based Aptitude questions under any circumstance.
       - ALWAYS generate complete, self-contained, and fully detailed questions. NEVER generate truncated or incomplete questions.
       - MUST follow this structure:
         1. `"scenario"`: Full real-world / business context containing all necessary numerical values and parameters.
         2. `"question"`: Explicit calculation task asking what the candidate must calculate.
         3. `"problemStatement"`: Full combined problem statement (Scenario + Task).
         4. `"placeholder"`: `"Enter your answer"`
           * `"outputFormat"`: Analyze `expectedAnswer` FIRST, then derive format instructions stating the exact expected type (e.g., 'Output Format: Return an integer.', 'Output Format: Return a decimal.', 'Output Format: Return a percentage.', 'Output Format: Return a ratio.', 'Output Format: Return a time duration.', 'Output Format: Return a currency amount.', or 'Output Format: Return a boolean.').
         * `"answerType"`: `"NUMBER"` or `"TEXT"`
         * `"expectedAnswer"` and `"correctAnswer"`: exact expected numerical/text result string
         * `"explanation"`: detailed step-by-step calculation or reasoning.
       - NEVER generate incomplete questions such as returning only a scenario description without the actual calculation task request.
   - DIFFICULTY CALIBRATION FOR APTITUDE:
     * Easy: Basic formula-based questions (single step calculations).
     * Medium: Requires two or more calculation steps.
     * Hard: Complex multi-step reasoning or multi-concept problems.
   - NOVELTY & VARIATION: Vary numbers, names, contexts, and wording on every generation so no two assessment questions are identical.

5. SCENARIO-BASED PYTHON CODING QUESTIONS:
   - STRICT MANDATE FOR REAL-WORLD SCENARIOS: Every Python coding question MUST be framed as a real-world scenario (e.g., School, College, Library, Hospital, Railway, Airport, Cricket, Restaurant, Shopping Mall, Parking, Employee Salary, Student Marks, Banking, Delivery, Weather, Attendance, Electricity Bill, Mobile Recharge, Hotel, Supermarket, Inventory, Online Shopping, Movie Ticket Booking, Examination, Bus Reservation).
   - BAN ON PLAIN TEXTBOOK QUESTIONS: Never generate plain textbook or generic algorithm questions without business context. Always wrap every question into a real-world scenario.
   - ALLOWED BEGINNER / INTERMEDIATE TOPICS: Generate ONLY from beginner or int    - DYNAMIC FUNCTION SIGNATURE MANDATE: Every Python question MUST specify a descriptive function name and parameter list matching the task (e.g., Write a Python function `filter_even(numbers)` or `calculate_bonus(salary, percentage)`). NEVER default to generic function names like `solution`, `solve`, or `process` unless explicitly required by the business scenario.
   - SAMPLE TEST CASE ONLY: Generate exactly ONE visible Sample Input (`sampleInput`) and Sample Output (`sampleOutput`).

6. TOPIC RELEVANCE:
   - Every question's "subject" MUST strictly be one of: {request.subjects}.

RESPONSE SCHEMA (RETURN RAW CLEAN JSON ONLY):
{{
  "questions": [
    {{
      "subject": "<String: Subject requested>",
      "topic": "<String: Topic generated dynamically>",
      "type": "MCQ",
      "difficulty": "<String: Easy|Medium|Hard>",
      "question": "<String: Dynamically generated MCQ question statement>",
      "options": ["<Option A>", "<Option B>", "<Option C>", "<Option D>"],
      "correctAnswer": "<String: Exactly one correct option choice matching an item in options>",
      "explanation": "<String: Detailed step-by-step reasoning or explanation>"
    }},
    {{
      "subject": "<String: Subject requested>",
      "topic": "<String: Topic generated dynamically>",
      "type": "SCENARIO",
      "difficulty": "<String: Easy|Medium|Hard>",
      "scenario": "<String: Dynamically generated real-world/business context>",
      "question": "<String: Explicit task instructions generated dynamically by AI>",
      "task": "<String: Explicit task instructions generated dynamically by AI>",
      "problemStatement": "### Scenario:\\n<Scenario text>\\n\\n---\\n\\n### Task:\\n<Task text>\\n\\n---\\n\\n### Input/Output Format:\\n<Input/Output Format text>\\n\\n---\\n\\n### Example:\\nInput: <Sample Input>\\nOutput: <Sample Output>",
      "candidateTask": "<String: Short task summary>",
      "starterCode": "def <dynamic_func_name>(<params>):\\n    pass",
      "sampleInput": "<String: Valid sample input>",
      "sampleOutput": "<String: Valid sample output>",
      "expectedAnswer": "<String: Dynamic solution query or code>",
      "explanation": "<String: Detailed step-by-step explanation>"d step-by-step explanation>"
    }}
  ]
}}"""
        return prompt

    def _normalize_and_validate_sql_question(self, q: dict) -> dict:
        """
        Validates and normalizes SQL scenario questions to enforce 100% adherence
        to strict readability rules:
        - Scenario: 2-4 sentences business scenario
        - Task: Short sentences, bold table names (**Schema.Table**) with purpose in parentheses on separate lines,
                followed by bulleted requirements for column selection, joins, filtering, grouping, and sorting.
        - Input/Output Format: Clean bulleted list of output columns
        - ProblemStatement: Standardized 3-part layout with '---' section dividers
        """
        import re

        scen_raw = str(q.get("scenario") or "").strip()
        task_raw = str(q.get("task") or q.get("candidateTask") or "").strip()
        prob_raw = str(q.get("problemStatement") or q.get("question") or "").strip()
        expected_query = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()

        # 1. Clean Scenario
        clean_scenario = scen_raw
        if not clean_scenario or len(clean_scenario) < 15:
            if "### Scenario:" in prob_raw:
                parts = prob_raw.split("---")
                clean_scenario = parts[0].replace("### Scenario:", "").strip()
            else:
                clean_scenario = f"A real-world database analysis scenario evaluating {q.get('topic', 'SQL query logic')}."
        
        for p in ["Scenario:", "### Scenario:", "---"]:
            clean_scenario = clean_scenario.replace(p, "").strip()

        # 2. Format Task section into clean table usage lines + bulleted requirements
        raw_task_text = task_raw
        if not raw_task_text or len(raw_task_text) < 15:
            if "### Task:" in prob_raw:
                parts = prob_raw.split("---")
                if len(parts) >= 2:
                    raw_task_text = parts[1].replace("### Task:", "").strip()

        # Split raw task into sentences or lines
        raw_lines = [l.strip() for l in raw_task_text.replace("\r", "").split("\n") if l.strip()]
        if len(raw_lines) == 1 and "." in raw_lines[0]:
            raw_lines = [s.strip() + "." for s in raw_lines[0].split(".") if s.strip()]

        table_lines = []
        requirement_bullets = []

        schemas = ["HumanResources", "Sales", "Production", "Purchasing", "Person", "dbo"]

        for s in raw_lines:
            s_clean = s.strip()
            if not s_clean or any(h in s_clean for h in ["###", "Scenario:", "Task:", "Input/Output", "---"]):
                continue

            has_table = any(f"{schema}." in s_clean for schema in schemas) or re.search(r'\b[A-Z][a-zA-Z0-9_]+\.[A-Z][a-zA-Z0-9_]+\b', s_clean)
            
            if has_table:
                # Bold table names e.g. Production.Product -> **Production.Product**
                table_formatted = re.sub(r'(?<!\*\*)(\b[A-Z][a-zA-Z0-9_]+\.[A-Z][a-zA-Z0-9_]+\b)(?!\*\*)', r'**\1**', s_clean)
                
                # Ensure starts with "Use " if not starting with action verb or bullet
                if not any(table_formatted.startswith(prefix) for prefix in ["Use ", "Query ", "Select ", "From ", "Join ", "- ", "* "]):
                    table_formatted = f"Use {table_formatted}"
                
                table_lines.append(table_formatted)
            else:
                if s_clean.startswith("- ") or s_clean.startswith("* "):
                    requirement_bullets.append(s_clean if s_clean.startswith("- ") else f"- {s_clean[2:]}")
                else:
                    requirement_bullets.append(f"- {s_clean.rstrip('.')}.")

        final_task_parts = []
        if table_lines:
            final_task_parts.extend(table_lines)
            final_task_parts.append("")  # Empty line separator

        if requirement_bullets:
            final_task_parts.extend(requirement_bullets)
        else:
            final_task_parts.append(f"- Query the specified database tables and return the required columns.")

        clean_task = "\n".join(final_task_parts).strip()

        # 3. Extract output columns from expected query for bulleted Input/Output Format
        output_cols = []
        if expected_query:
            m = re.search(r'SELECT\s+(?:TOP\s+\d+\s+)?(.*?)\s+FROM', expected_query, re.IGNORECASE | re.DOTALL)
            if m:
                col_exprs = m.group(1).split(",")
                for expr in col_exprs:
                    expr_clean = expr.strip()
                    if " AS " in expr_clean.upper():
                        alias = re.split(r'\s+AS\s+', expr_clean, flags=re.IGNORECASE)[-1].strip()
                        alias_clean = alias.replace("[", "").replace("]", "").strip()
                        if alias_clean:
                            output_cols.append(alias_clean)
                    else:
                        parts = expr_clean.split(".")
                        col_name = parts[-1].split()[-1].replace("[", "").replace("]", "").strip()
                        if col_name and not col_name.upper().startswith("TOP"):
                            output_cols.append(col_name)

        if not output_cols:
            output_cols = ["Column1", "Column2", "Column3"]

        bulleted_cols = "\n".join([f"- {col}" for col in output_cols])
        clean_io_fmt = (
            f"Query the live database and return these columns:\n\n"
            f"{bulleted_cols}"
        )

        # 4. Standardized Problem Statement
        prob_stmt = (
            f"### Scenario:\n"
            f"{clean_scenario}\n\n"
            f"---\n\n"
            f"### Task:\n"
            f"{clean_task}\n\n"
            f"---\n\n"
            f"### Input/Output Format:\n"
            f"{clean_io_fmt}"
        )

        # Strip unrequested TOP/LIMIT clauses from expected query if task does not explicitly ask for row limits
        task_scen_combined = f"{clean_scenario} {clean_task}".lower()
        explicit_limit_keywords = [
            "top 5", "top 10", "top 3", "top 20", "top ",
            "limit 5", "limit 10", "limit 3", "limit 20", "limit ",
            "first 5", "first 10", "first 3", "first 20",
            "highest 5", "highest 10", "lowest 5", "lowest 10",
            "only 5", "only 10", "only 3"
        ]
        asks_for_limit = any(k in task_scen_combined for k in explicit_limit_keywords)
        if expected_query and not asks_for_limit:
            expected_query = re.sub(r'SELECT\s+TOP\s*\(?\s*\d+\s*\)?\s+', 'SELECT ', expected_query, flags=re.IGNORECASE)
            expected_query = re.sub(r'\s+LIMIT\s+\d+\s*;?$', ';', expected_query, flags=re.IGNORECASE)
            expected_query = re.sub(r'\s+FETCH\s+(?:FIRST|NEXT)\s+\d+\s+ROWS?\s+ONLY\s*;?$', ';', expected_query, flags=re.IGNORECASE)
            q["expectedAnswer"] = expected_query
            q["correctAnswer"] = expected_query

        q["scenario"] = clean_scenario
        q["task"] = clean_task
        q["candidateTask"] = clean_task
        q["inputOutputFormat"] = clean_io_fmt
        q["problemStatement"] = prob_stmt
        q["question"] = prob_stmt

        return q

    def _normalize_and_validate_python_question(self, q: dict) -> dict:
        """
        Validates and normalizes Python scenario questions to enforce 100% consistency
        between problem statement, function signature, parameter naming, input format,
        sample input, starter code, and visible test cases.
        Dynamically preserves AI-generated parameter names and signatures.
        """
        topic_lower = str(q.get("topic", "")).lower()
        title_lower = str(q.get("title", "")).lower()
        scen_raw = str(q.get("scenario") or "").strip()
        task_raw = str(q.get("task") or q.get("candidateTask") or "").strip()
        prob_raw = str(q.get("problemStatement") or q.get("question") or "").strip()
        q_full_text = f"{title_lower} {topic_lower} {scen_raw} {task_raw} {prob_raw}".lower()

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

        # 1. Dynamically extract function name and parameter names from AI generated output
        extracted_func_name = None
        extracted_params = []

        banned_kws = {"function", "python", "def", "write", "takes", "returns", "given", "that", "and", "a", "an", "the", "self", "pass", "solution", "solve", "process", "main"}

        sources = [
            task_raw,
            prob_raw,
            str(q.get("functionSignature") or ""),
            str(q.get("starterCode") or q.get("starter_code") or "")
        ]

        for src in sources:
            if not src:
                continue
            
            # Match `def func_name(param1, param2)`
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", src)
            if not match:
                match = re.search(r"function\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s*\((.*?)\)", src, re.IGNORECASE)
            if not match:
                match = re.search(r"`([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)`", src)
            if not match:
                match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([a-zA-Z0-9_,\s]*)\)", src)

            if match:
                fn = match.group(1).strip()
                # Keep AI's exact function name if specified in Task
                if fn.isidentifier() and fn.lower() not in {"function", "python", "def", "write", "takes", "returns", "given", "that", "and", "a", "an", "the", "self", "pass"}:
                    extracted_func_name = fn

                raw_p = match.group(2).strip()
                if raw_p:
                    params_list = [p.strip().split(":")[0].strip().split("=")[0].strip() for p in raw_p.split(",") if p.strip()]
                    clean_p = [p for p in params_list if p and p != "self" and p.isidentifier() and p.lower() not in {"self", "pass"}]
                    if clean_p:
                        extracted_params = clean_p
                
                if extracted_func_name and extracted_params:
                    break

        # Fallbacks if function name or params were not explicitly provided
        if not extracted_func_name:
            raw_topic = str(q.get("topic") or q.get("title") or "").lower()
            clean_words = [w for w in re.sub(r"[^a-zA-Z0-9_\s]", "", raw_topic).split() if w not in banned_kws and len(w) > 2]
            if clean_words:
                extracted_func_name = "_".join(clean_words[:3])
                if not extracted_func_name.isidentifier():
                    extracted_func_name = f"func_{extracted_func_name}"
            else:
                extracted_func_name = "solution"

        if not extracted_params:
            if any(w in q_full_text for w in ["weight", "parcel"]):
                extracted_params = ["weights"]
            elif any(w in q_full_text for w in ["price", "cart", "cost"]):
                extracted_params = ["prices"]
            elif any(w in q_full_text for w in ["score", "mark", "student"]):
                extracted_params = ["scores"]
            elif any(w in q_full_text for w in ["temp", "weather"]):
                extracted_params = ["temperatures"]
            elif any(w in q_full_text for w in ["text", "string", "word"]):
                extracted_params = ["text"]
            elif "matrix" in q_full_text or "grid" in q_full_text:
                extracted_params = ["matrix"]
            else:
                extracted_params = ["numbers"]

        func_sig = f"def {extracted_func_name}({', '.join(extracted_params)}):"
        starter_code = f"{func_sig}\n    pass"

        clean_scenario = scen_raw if len(scen_raw) >= 15 else (prob_raw[:200] if len(prob_raw) >= 15 else f"A real-world operational scenario evaluating {q.get('topic', 'Python logic')}.")
        for p in ["Scenario:", "Task:", "Input Format:", "Output Format:", "Input/Output Format:", "Example:", "Input:", "Output:"]:
            clean_scenario = re.sub(rf'^{p}\s*', '', clean_scenario, flags=re.IGNORECASE).strip()

        # Format Task to guarantee explicit mention of the exact function name and parameters
        if task_raw:
            clean_task = task_raw
            if extracted_func_name not in clean_task:
                clean_task = f"Write a Python function `{extracted_func_name}({', '.join(extracted_params)})` that takes {', '.join(extracted_params)} and returns the calculated result.\n\n{task_raw}"
        else:
            clean_task = f"Write a Python function `{extracted_func_name}({', '.join(extracted_params)})` to process the input parameters and return the expected output."

        in_fmt = str(q.get("inputFormat") or q.get("inputOutputFormat") or "").strip()
        if not in_fmt:
            in_fmt = f"Input parameters: {', '.join(extracted_params)}"

        out_fmt = str(q.get("outputFormat") or "").strip()
        if not out_fmt:
            out_fmt = self._infer_explicit_output_format(q_full_text, raw_out)

        io_fmt_str = f"Input parameters: {in_fmt}" if "input is" not in in_fmt.lower() else in_fmt

        formatted_q = (
            f"### Scenario:\n"
            f"{clean_scenario}\n\n"
            f"---\n\n"
            f"### Task:\n"
            f"{clean_task}\n\n"
            f"---\n\n"
            f"### Input Format:\n"
            f"{in_fmt}\n\n"
            f"---\n\n"
            f"### Example:\n"
            f"Input: {raw_in}\n\n"
            f"Output: {raw_out}"
        )

        q["functionSignature"] = func_sig
        q["starterCode"] = starter_code
        q["starter_code"] = starter_code
        q["inputFormat"] = in_fmt
        q["outputFormat"] = None
        q["scenario"] = clean_scenario
        q["task"] = clean_task
        q["candidateTask"] = clean_task
        q["problemStatement"] = formatted_q
        q["question"] = formatted_q
        q["sampleInput"] = raw_in
        q["sampleOutput"] = raw_out
        q["exampleInput"] = raw_in
        q["exampleOutput"] = raw_out

        visible_tc = {"input": raw_in, "expectedOutput": raw_out, "output": raw_out}
        q["visibleTestCase"] = visible_tc
        q["visibleTestCases"] = [visible_tc]
        q["hiddenTestCases"] = []
        q["testCases"] = {
            "visible": [{"input": raw_in, "output": raw_out}],
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
        Dynamically analyzes expectedAnswer and returns ONLY the concise answer data type string
        (e.g., Integer, Decimal, Percentage, Fraction, Ratio, Time, Currency, Boolean, String).
        No extra instructions, rules, or explanations.
        """
        import re
        topic_lower = (topic or "").lower()
        exp_str = (raw_expected or "").strip()

        # Clean existing_fmt if it already specifies a concise type
        if existing_fmt and isinstance(existing_fmt, str) and existing_fmt.strip():
            clean_fmt = existing_fmt.strip()
            if "Output Format:" in clean_fmt:
                clean_fmt = clean_fmt.split("Output Format:")[-1].split("\n")[0].strip()
            clean_fmt = re.sub(r'^[-\*\s\n\•]+', '', clean_fmt).split('\n')[0].strip()
            clean_fmt = re.sub(r'^(Return a|Return an|Return)\s+', '', clean_fmt, flags=re.IGNORECASE).strip()
            if clean_fmt and len(clean_fmt) < 25 and not any(kw in clean_fmt.lower() for kw in ["enter", "round", "symbol", "do not", "must"]):
                return clean_fmt.capitalize(), exp_str

        exp_clean = exp_str.replace(",", "").strip()

        # 1. Boolean expectedAnswer
        if exp_clean.lower() in ["true", "false", "yes", "no"]:
            return "Boolean", exp_clean.capitalize()

        # 2. Fraction expectedAnswer (e.g. "3/4", "1/2")
        fraction_match = re.search(r'\b\d+\s*/\s*\d+\b', exp_clean)
        if fraction_match:
            return "Fraction", fraction_match.group(0).replace(" ", "")

        # 3. Ratio expectedAnswer (e.g. "3:2", "5:4")
        ratio_match = re.search(r'\b\d+\s*:\s*\d+\b', exp_clean)
        if ratio_match or (":" in exp_clean and any(c.isdigit() for c in exp_clean)) or "ratio" in topic_lower or "proportion" in topic_lower:
            clean_exp = ratio_match.group(0).replace(" ", "") if ratio_match else exp_clean
            return "Ratio", clean_exp

        # 4. Percentage expectedAnswer (%)
        is_pct_symbol = "%" in exp_str
        is_pct_topic = any(kw in topic_lower for kw in ["percentage", "interest", "profit and loss", "discount", "margin", "probability"])
        is_curr_symbol = any(sym in exp_str for sym in ["$", "₹", "€", "£"])
        is_curr_topic = any(kw in topic_lower for kw in ["cost", "price", "salary", "partnership", "investment", "amount", "money"])
        is_time_unit = any(unit in exp_str.lower() for unit in ["day", "hour", "minute", "second", "year", "month"])
        is_time_topic = any(kw in topic_lower for kw in ["work", "speed", "distance", "cistern", "pipe", "clock", "calendar", "age", "time"])

        if is_pct_symbol or (is_pct_topic and not is_curr_symbol and not is_curr_topic and not is_time_unit and not is_time_topic and any(c.isdigit() for c in exp_str)):
            clean_exp = re.sub(r'[\$,₹,€,£,%,]', '', exp_str).strip()
            try:
                val = float(clean_exp)
                clean_exp = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
            except ValueError:
                pass
            return "Percentage", clean_exp

        # 5. Currency expectedAnswer ($ ₹ € £)
        if is_curr_symbol or (is_curr_topic and not is_time_unit and not is_time_topic and any(c.isdigit() for c in exp_str)):
            clean_exp = re.sub(r'[\$,₹,€,£,]', '', exp_str).strip()
            try:
                val = float(clean_exp)
                clean_exp = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else str(int(val))
            except ValueError:
                pass
            return "Currency", clean_exp

        # 6. Time / Duration expectedAnswer
        if is_time_unit or (is_time_topic and any(c.isdigit() for c in exp_str)):
            clean_exp = exp_str.replace(",", "").strip()
            return "Time", clean_exp

        # 7. Numeric expectedAnswer: Integer vs Decimal
        if any(c.isdigit() for c in exp_str):
            clean_exp = re.sub(r'[^\d\.]', '', exp_str).strip()
            if "." in clean_exp:
                try:
                    val = float(clean_exp)
                    if val % 1 != 0:
                        clean_exp = f"{val:.2f}".rstrip('0').rstrip('.')
                        return "Decimal", clean_exp
                    else:
                        clean_exp = str(int(val))
                        return "Integer", clean_exp
                except ValueError:
                    pass
                return "Decimal", clean_exp
            else:
                return "Integer", clean_exp

        # 8. String (Text)
        return "String", exp_str

    def _clean_and_validate_questions(
        self, 
        questions: List[Dict[str, Any]], 
        target_subjects: List[str],
        existing_questions: Optional[List[str]] = None
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

                elif q["subject"].upper() == "SQL":
                    q = self._normalize_and_validate_sql_question(q)
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
                if q["subject"].upper() == "PYTHON":
                    q["outputFormat"] = None
                elif not q.get("outputFormat"):
                    if q["subject"].upper() == "APTITUDE":
                        raw_exp = str(q.get("expectedAnswer") or q.get("correctAnswer") or "0")
                        raw_top = str(q.get("topic") or "Aptitude")
                        derived_fmt, _ = self._derive_aptitude_output_format(raw_exp, raw_top)
                        q["outputFormat"] = derived_fmt
                    else:
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
                        q["databaseSchema"] = SqlSchemaService.get_database_schemas_for_question(q, tables_map)

                # Final strict validation and similarity check for Python and SQL scenario questions
                if q["type"] == "SCENARIO":
                    if q["subject"].upper() == "PYTHON":
                        if not validate_python_question(q):
                            logger.warning(f"[AzureOpenAIService] Discarding invalid Python question '{q.get('topic')}'.")
                            continue
                        if is_duplicate_or_similar(q, existing_questions, cleaned_questions):
                            logger.warning(f"[AzureOpenAIService] Discarding duplicate/similar Python question '{q.get('topic')}'.")
                            continue
                    elif q["subject"].upper() == "SQL":
                        if not validate_sql_question(q):
                            logger.warning(f"[AzureOpenAIService] Discarding invalid SQL question '{q.get('topic')}'.")
                            continue
                        if is_duplicate_or_similar(q, existing_questions, cleaned_questions):
                            logger.warning(f"[AzureOpenAIService] Discarding duplicate/similar SQL question '{q.get('topic')}'.")
                            continue

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

        cat_upper = str(request.category).upper()
        top_upper = str(request.topic).upper()
        if "SQL" in cat_upper or "SQL" in top_upper:
            feature_name = AiFeature.SQL_QUESTION_GENERATION
        elif "PYTHON" in cat_upper or "PYTHON" in top_upper:
            feature_name = AiFeature.PYTHON_QUESTION_GENERATION
        else:
            feature_name = AiFeature.ASSESSMENT_EVALUATION

        raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=feature_name)
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

        cat_upper = str(request.category).upper()
        top_upper = str(request.topic).upper()
        if "SQL" in cat_upper or "SQL" in top_upper:
            feature_name = AiFeature.SQL_ANSWER_EVALUATION
        elif "PYTHON" in cat_upper or "PYTHON" in top_upper:
            feature_name = AiFeature.PYTHON_ANSWER_EVALUATION
        else:
            feature_name = AiFeature.ASSESSMENT_EVALUATION

        raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=feature_name)
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

        q_upper = str(question).upper()
        ans_upper = str(correct_answer).upper()
        if "SQL" in q_upper or "SELECT" in ans_upper or "FROM" in ans_upper:
            feature_name = AiFeature.SQL_ANSWER_EVALUATION
        elif "PYTHON" in q_upper or "DEF " in ans_upper or "PRINT" in ans_upper:
            feature_name = AiFeature.PYTHON_ANSWER_EVALUATION
        else:
            feature_name = AiFeature.ASSESSMENT_EVALUATION

        raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=feature_name)
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
            raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=AiFeature.CANDIDATE_FEEDBACK_GENERATION)
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
        
        raw_response = await self.client.generate_chat_completion(prompt, system_message, feature_name=AiFeature.RESUME_ANALYSIS)
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

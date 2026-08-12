import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger("recruitai-backend.gemini_service")

class GeminiService:
    def __init__(self):
        self.api_key = getattr(settings, "GEMINI_API_KEY", "")
        if not self.api_key:
            import os
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        
        # For server-side REST API calls, use a text-capable model.
        # If the main GEMINI_MODEL contains "live", it is a WebSocket-only model
        # and cannot be used for generateContent. We fall back to GEMINI_TEXT_MODEL.
        model_setting = getattr(settings, "GEMINI_TEXT_MODEL", "") or getattr(settings, "GEMINI_MODEL", "gemini-3.1-flash-lite")
        if "live" in model_setting.lower():
            self.model = "gemini-3.1-flash-lite"
        else:
            self.model = model_setting
            
        self.voice = getattr(settings, "GEMINI_VOICE", "Puck")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    async def _call_ai(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Executes AI completion using the configured Gemini model (gemini-3.1-flash-lite).
        """
        if self.api_key:
            model_name = self.model
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ]
                }
                if system_instruction:
                    payload["systemInstruction"] = {
                        "parts": [{"text": system_instruction}]
                    }
                if json_mode:
                    payload["generationConfig"] = {"responseMimeType": "application/json"}

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, headers=headers, timeout=10.0)
                    if response.status_code == 200:
                        data = response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text.strip()
                    else:
                        logger.error(f"Gemini API ({model_name}) returned error status {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"Gemini API ({model_name}) call failed: {e}")

        raise ValueError("Gemini API call failed and no alternative AI model is configured.")

    async def generate_first_question(self, candidate_name: str, resume_text: str) -> str:
        """
        Generates a warm, professional greeting and introductory request.
        """
        system_instruction = (
            "You are a warm, professional corporate HR Manager conducting a real-time voice interview. "
            "Your tone is warm, supportive, and conversational. "
            "CRITICAL: You must greet the candidate and ask them to tell you about themselves. "
            "Keep the response strictly under 25 words (1 to 2 short sentences) so it is smooth for Text-To-Speech (TTS)."
        )

        prompt = f"""
        Candidate Name: {candidate_name}
        Candidate Resume Information:
        \"\"\"
        {resume_text}
        \"\"\"

        Task:
        1. Greet the candidate using their first name.
        2. If the resume contains projects or skills, generate a personalized, resume-aware opening question (e.g. 'Hi {candidate_name}, nice to meet you! Could you tell me about yourself and your work on [Project/Skill] listed on your resume?').
        3. If no resume details exist, output: 'Hi {candidate_name}, nice to meet you! Could you please tell me about yourself and give a brief overview of your background?'
        4. Keep the greeting strictly under 25 words total.
        """
        try:
            return await self._call_ai(prompt, system_instruction)
        except Exception as e:
            logger.warning(f"Failed to generate first question via AI API: {e}. Using personalized fallback.")
            return f"Hi {candidate_name}, nice to meet you! Could you please tell me about yourself and give a brief overview of your background?"

    async def generate_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Analyzes candidate's last response and resume details to generate the next personalized interview question or follow-up question.
        Returns a JSON with analysis notes and the next question string.
        """
        # Verbatim repeat command check
        last_clean = last_answer.lower().strip().replace(".", "").replace("?", "").replace("!", "")
        repeat_phrases = [
            "repeat the question", "please repeat", "repeat please", "can you repeat", "could you repeat",
            "say that again", "could you say that again", "please say that again"
        ]
        if any(p == last_clean for p in repeat_phrases) and conversation_history:
            prev_q = conversation_history[-1].get("ai_question")
            if prev_q:
                return {
                    "analysis": {
                        "grammar_notes": "Candidate requested to repeat the question.",
                        "fluency_notes": "N/A",
                        "confidence_notes": "N/A"
                    },
                    "next_question": prev_q
                }

        system_instruction = (
            "You are a warm, empathetic human corporate HR Manager conducting a voice assessment. "
            "Your tone must be warm, professional, engaging, and completely conversational, like a real human interviewer on a phone call. "
            "Acknowledge the candidate's last response briefly and naturally (e.g., 'That is interesting!', 'Great work!', 'I see, that makes sense') "
            "before transitioning to your next question. Do NOT sound like an AI assistant or a robotic system. "
            "CRITICAL: Keep the next question direct, natural, conversational, and strictly under 25 words so it is voice-synthesis ready. "
            "You must respond ONLY with a valid JSON object matching the requested schema. "
            "You must ask exactly ONE question at a time."
        )

        formatted_history = []
        for msg in conversation_history:
            formatted_history.append(f"AI Question: {msg.get('ai_question')}")
            formatted_history.append(f"Candidate Answer: {msg.get('candidate_answer')}")
        history_str = "\n".join(formatted_history)

        q_num = len(conversation_history) + 1

        prompt = f"""
        Candidate Resume Information:
        \"\"\"
        {resume_text}
        \"\"\"

        Conversation History So Far:
        {history_str}

        Latest Candidate Response to Question {q_num - 1}:
        \"{last_answer}\"

        Task:
        1. Classify the Candidate's Intent and generate Question {q_num}:
           - Case A (Repeat or Clarification Request): If the candidate says 'I didn't understand', 'Can you repeat?', 'pardon', 'clarify', or asks what a term means:
             * Acknowledge naturally (e.g., 'Sure, no problem.', 'No worries!') and repeat or explain the SAME question simply. Do NOT skip to a new topic.
           - Case B (Question related to current context): Answer concisely first, then ask the same interview question again.
           - Case C (Unrelated Question / off-topic): Politely transition back using a natural HR transition and ask the next resume-based question.
           - Case D (Valid Answer): Respond to their answer with a brief, natural acknowledgment. Ask a progressive, relevant question focusing strictly on their background.

        2. IMPLEMENT RECRUITAI INTERVIEW PLANNING ENGINE:
           - STEP 1 — UNDERSTAND THE RESUME: Extract and organize the resume sections internally: Profile, Projects, Work Experience, Responsibilities, Achievements, Technical Skills (Programming Languages, Frameworks, Databases, Cloud, DevOps, Testing, Tools), Certifications, Education, Internships, Leadership, Awards, Open Source, Research, Publications. If a section is missing, ignore it. Never invent missing details.
           - STEP 2 — BUILD AN INTERVIEW PLAN: Dynamic interview plan tracking Topic, Coverage, Questions Asked, Remaining Questions, Competencies Covered, Current Difficulty, Previous Follow-up Count, and Current Interview Stage.
           - STEP 3 — CHOOSE THE NEXT TOPIC: Least explored section, project not discussed, competency needing evaluation, skill not assessed, achievement needs clarification, responsibility deserving deeper exploration. Always choose the highest-value uncovered topic.
           - STEP 4 — PROJECT QUESTIONS: If projects exist, ask about problem solved, role, responsibilities, tech choices, architecture, challenges, debugging, optimization, deployment, performance, scalability, lessons, business impact. Ask clarification if info is missing.
           - STEP 5 — SKILL QUESTIONS: Ask practical implementation questions (e.g., FastAPI selection, SQL optimization, auth implementation, debugging challenges). Avoid textbook questions (e.g., "What is Python?", "Define REST API").
           - STEP 6 — EXPERIENCE QUESTIONS: Ask about responsibilities, ownership, decision making, collaboration, problem solving, delivery, learning, production issues, impact.
           - STEP 7 — ACHIEVEMENT QUESTIONS: If achievements exist, ask how success was measured, how achieved, challenges overcome, contribution, and learning.
           - STEP 8 — CLARIFICATION RULE: If resume contains incomplete or ambiguous information, ask ONE clarification question (e.g. chatbot details). Do not assume technologies/platforms not listed. Limit follow-up to maximum one, then move on.
           - STEP 9 — RESPONSE ANALYSIS: Classify answer as Excellent, Strong, Average, Weak, Incomplete, Irrelevant, or Silence. Excellent -> Architecture/Trade-offs/Optimization; Strong -> Implementation; Average -> Practical examples; Weak -> Simpler question; Incomplete -> One clarification; Irrelevant -> Redirect; Silence -> Encourage then continue.
           - STEP 10 — DUPLICATE PREVENTION: Compare potential question against all past questions to avoid semantic duplicates (e.g. React project vs React application). Never repeat intent.
           - STEP 11 — HALLUCINATION PREVENTION: Resume is the only source of truth. Never invent projects, companies, responsibilities, cloud providers, deployment, tech, team size, metrics. If unsure, ask clarification.
           - STEP 12 — QUESTION QUALITY: Must be resume-based, natural, conversational, under 25 words, specific, practical, competency-driven, unique, relevant, and human-like.
           - STEP 13 — RESPONSE FORMAT: Return ONLY 'analysis', 'is_relevant', and 'next_question'.

        You MUST return a valid JSON object matching this schema:
        {{
          "internal_thought": {{
            "resume_profile": {{
              "profile": {{}},
              "projects": [],
              "work_experience": [],
              "technical_skills": {{}}
            }},
            "interview_plan": {{
              "topic": "Projects",
              "coverage": "0%",
              "questions_asked": 0,
              "remaining_questions": 3,
              "competencies_covered": [],
              "current_difficulty": "Strong",
              "previous_followup_count": 0,
              "current_interview_stage": "Intro"
            }},
            "response_analysis": "Average",
            "verification_checklist": {{
              "resume_based": true,
              "natural": true,
              "under_25_words": true,
              "specific": true,
              "practical": true,
              "competency_driven": true
            }}
          }},
          "analysis": {{
            "grammar_notes": "A brief sentence summarizing grammar performance.",
            "fluency_notes": "A brief sentence summarizing fluency and coherence.",
            "confidence_notes": "A brief sentence summarizing confidence and vocabulary.",
            "relevance_notes": "A brief sentence evaluating whether the candidate's latest response was relevant."
          }},
          "is_relevant": true,
          "next_question": "The exact single response or interview question to present to the candidate."
        }}
        """

        try:
            response_text = await self._call_ai(prompt, system_instruction, json_mode=True)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            # Return only is_relevant, next_question, and analysis to comply with API contracts
            return {
                "analysis": data.get("analysis", {}),
                "is_relevant": data.get("is_relevant", True),
                "next_question": data.get("next_question", "")
            }
        except Exception as e:
            logger.warning(f"Failed to generate next question via AI API: {e}. Using personalized fallback.")
            return self._get_fallback_next_question(resume_text, conversation_history, last_answer)

    def _get_fallback_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Resume-aware fallback question generator when AI API service is unavailable.
        Parses resume programmatically, rotates topics, prevents semantic duplicates, and stays under 25 words.
        """
        q_count = len(conversation_history) + 1
        low_answer = last_answer.lower().strip()

        # Handle repeat fallback
        repeat_phrases = ["repeat", "pardon", "say again", "didn't catch", "didnt catch", "what did you say", "what was the question"]
        if any(p in low_answer for p in repeat_phrases) and conversation_history:
            prev_q = conversation_history[-1].get("ai_question")
            if prev_q:
                return {
                    "analysis": {
                        "grammar_notes": "Candidate requested to repeat the question.",
                        "fluency_notes": "N/A",
                        "confidence_notes": "N/A",
                        "relevance_notes": "Requested question repetition."
                    },
                    "is_relevant": True,
                    "next_question": prev_q
                }

        # Handle unrelated/don't know answer fallback
        unrelated_phrases = ["don't know", "dont know", "no idea", "skip", "pass", "not sure"]
        is_unrelated = any(p in low_answer for p in unrelated_phrases)
        transition_prefix = "No problem! Let's move on. " if is_unrelated else ""

        # Extract details programmatically
        common_tech = ["Python", "JavaScript", "React", "SQL", "PostgreSQL", "FastAPI", "AWS", "Docker", "Kubernetes", "Git", "Java", "C++", "HTML", "CSS", "Tailwind", "Node.js", "Django", "Flask"]
        found_techs = [tech for tech in common_tech if tech.lower() in resume_text.lower()]
        
        lines = [line.strip() for line in resume_text.split("\n") if len(line.strip()) > 5]
        projects = []
        companies = []
        educations = []
        certifications = []
        achievements = []
        internships = []

        for line in lines:
            line_low = line.lower()
            if any(kw in line_low for kw in ["intern at", "internship at", "intern software", "co-op"]):
                parts = line.split(" at ")
                comp = parts[-1].split("(")[0].strip(".:,; ") if len(parts) > 1 else ""
                if len(comp) < 30 and comp and comp not in internships:
                    internships.append(comp)
            elif any(kw in line_low for kw in ["engineer at", "developer at", "analyst at", "work at"]):
                parts = line.split(" at ")
                comp = parts[-1].split("(")[0].strip(".:,; ") if len(parts) > 1 else ""
                if len(comp) < 30 and comp and comp not in companies:
                    companies.append(comp)
            elif "project" in line_low or "built" in line_low or "developed" in line_low or "created" in line_low:
                if len(line) < 80 and not any(kw in line_low for kw in ["information", "resume", "detail"]):
                    proj = line.split(":")[0].replace("-", "").strip(".:,; ")
                    if len(proj) < 50 and proj not in projects:
                        projects.append(proj)
            elif any(kw in line_low for kw in ["university", "college", "degree", "bachelor", "master"]):
                edu = line.split(",")[0].strip(".:,; ")
                if len(edu) < 50 and edu not in educations:
                    educations.append(edu)
            elif any(kw in line_low for kw in ["certified", "certification", "credential"]):
                cert = line.split(":")[-1].strip(".:,; ")
                if len(cert) < 50 and cert not in certifications:
                    certifications.append(cert)
            elif any(kw in line_low for kw in ["won", "award", "hackathon", "first place", "scholarship"]):
                if len(line) < 100:
                    achievements.append(line.strip(".:,; "))

        # Build prioritized question pool (Resume-First, under 25 words)
        # Priority order: Projects -> Experience -> Responsibilities -> Technical Skills -> Achievements -> Certifications -> Education -> Internships -> HR
        topics = []
        
        # 1. Projects
        for p in projects[:3]:
            topics.append(f"Regarding your project {p}, what was the main technical challenge you faced?")
            topics.append(f"How did you select the technologies for {p}?")
        
        # 2. Experience & Responsibilities
        for c in companies[:2]:
            topics.append(f"During your work experience at {c}, what were your primary responsibilities?")
            topics.append(f"What was your most proud achievement while working at {c}?")

        # 3. Technical Skills
        for t in found_techs[:3]:
            topics.append(f"How did you apply your {t} skills in your projects or experience?")

        # 4. Achievements
        for ach in achievements[:2]:
            # Limit description to under 15 words inside formatting
            ach_words = ach.split()
            ach_short = " ".join(ach_words[:6])
            topics.append(f"Could you share your contribution to the achievement '{ach_short}'?")

        # 5. Certifications
        for ct in certifications[:2]:
            topics.append(f"How has your {ct} certification helped you in practical software development?")

        # 6. Education
        for ed in educations[:1]:
            topics.append(f"How did your studies prepare you for a professional software development role?")

        # 7. Internships
        for intern in internships[:2]:
            topics.append(f"What was the most valuable technical lesson you learned during your internship at {intern}?")

        # Fallbacks (General HR/Behavioral under 25 words)
        topics.append("Could you describe a challenging situation you faced at work and how you handled it?")
        topics.append("How do you usually collaborate with team members on coding projects?")
        topics.append("Why are you interested in this role, and what motivates you?")
        topics.append("Where do you see your career path progressing over the next few years?")

        # Deduplicate programmatically against previous history
        previous_questions = [msg.get("ai_question", "").lower() for msg in conversation_history]
        filtered_topics = []
        for t in topics:
            t_low = t.lower()
            is_dup = False
            for pq in previous_questions:
                # Filter out topics sharing key names or significant overlap
                words_overlap = set(t_low.split()) & set(pq.split())
                sig_overlap = {w for w in words_overlap if len(w) > 4 and w not in {"could", "would", "about", "project", "experience"}}
                if len(sig_overlap) >= 2:
                    is_dup = True
                    break
            if not is_dup:
                filtered_topics.append(t)

        if not filtered_topics:
            filtered_topics = [
                "Could you describe a challenging situation you faced at work and how you handled it?",
                "How do you usually collaborate with team members on coding projects?",
                "Why are you interested in this role, and what motivates you?",
                "Where do you see your career path progressing over the next few years?"
            ]

        idx = (q_count - 1) % len(filtered_topics)
        next_q = transition_prefix + filtered_topics[idx]
        is_relevant = not is_unrelated

        return {
            "analysis": {
                "grammar_notes": "Good sentence structure.",
                "fluency_notes": "Fluent delivery.",
                "confidence_notes": "Exhibited professional confidence.",
                "relevance_notes": "Response is relevant/related to the question." if is_relevant else "Response is unrelated/off-topic."
            },
            "is_relevant": is_relevant,
            "next_question": next_q
        }

    async def generate_final_report(self, resume_text: str, conversation_history: List[Dict[str, str]], voice_used: bool = False) -> Dict[str, Any]:
        """
        Compiles the overall English Assessment Report evaluating candidate's responses against resume and linguistic standards.
        """
        system_instruction = (
            "You are an expert corporate HR senior evaluator and English Communication Evaluator. You must assess the entire conversation transcript "
            "and candidate's resume to generate a detailed, professional evaluation report. "
            "You must return ONLY a valid JSON object matching the requested schema."
        )

        formatted_history = []
        for msg in conversation_history:
            formatted_history.append(f"AI Question: {msg.get('ai_question')}")
            formatted_history.append(f"Candidate Answer: {msg.get('candidate_answer')}")
        history_str = "\n".join(formatted_history)

        prompt = f"""
        Candidate Resume Information:
        \"\"\"
        {resume_text}
        \"\"\"

        Full Interview Conversation History:
        {history_str}

        Voice Recording Audio Used: {voice_used}

        Task:
        Perform a comprehensive linguistic and English Communication evaluation of the candidate's answers.
        
        You must evaluate:
        1. Grammar accuracy: Tense consistency, syntax correctness, subject-verb agreement.
        2. Vocabulary range: Precision and appropriateness of technical/professional words.
        3. Sentence structure: Syntactic complexity, clauses, proper connectors/conjunctions.
        4. Fluency & pacing: Speech flow, speed estimation, hesitation filler words (like "um", "uh").
        5. Communication: Expression clarity, logical coherence, and response completeness.
        6. Confidence: Sentence assertiveness, clarity, professional tone.
        7. Technical communication: Clarity when conveying technical concepts from the resume.
        8. Professionalism: Business etiquette, structured responses.
        9. Listening & Understanding: Comprehension and direct alignment with what was asked.
        10. Response relevance: Addressing the question. If off-topic, gibberish, or empty, reduce scores to minimum.
        
        CRITICAL RULES FOR RELEVANCE AND SCORING:
        - Generate scores on a scale of 1 to 10.
        - Balance English quality with answer relevance in your scoring. Do not award high marks for continuous speaking alone if it is off-topic.
        - The overall English score must reflect average performance, scaled down if responses are irrelevant or off-topic. If all answers are off-topic/gibberish, Overall Score MUST be exactly 1.
        - Generate completely unique feedback. Avoid generic templates.

        Determine:
        - Strengths: A list of 3 key observations.
        - Weaknesses: A list of specific errors or gaps observed.
        - Areas for Improvement: A list of actionable recommendations.
        - Summary: A professional candidate overview and technical depth summary.
        - Hire Recommendation: Yes/No/Maybe with reasoning.
        - Recommended English Level: CEFR level (A1, A2, B1, B2, C1, or C2).
        
        Return ONLY a JSON object matching this schema:
        {{
          "grammar_score": 8,
          "vocabulary_score": 8,
          "sentence_structure_score": 8,
          "fluency_score": 8,
          "communication_score": 8,
          "confidence_score": 8,
          "technical_communication_score": 8,
          "professionalism_score": 8,
          "listening_understanding_score": 8,
          "response_relevance_score": 8,
          "overall_score": 8,
          "pronunciation_score": 8,
          "overall_level": "Very Good",
          "recommendation": "Very Good",
          "summary": "Detailed summary paragraph here...",
          "strengths": ["Strength 1", "Strength 2", "Strength 3"],
          "weaknesses": ["Weakness 1", "Weakness 2"],
          "areas_for_improvement": ["Area 1", "Area 2"],
          "hire_recommendation": "Yes with reason...",
          "recommended_english_level": "B2"
        }}
        """

        try:
            response_text = await self._call_ai(prompt, system_instruction, json_mode=True)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            report = json.loads(cleaned.strip())
            
            # Fill missing keys with sensible defaults
            defaults = {
                "communication_score": 8,
                "grammar_score": 8,
                "vocabulary_score": 8,
                "confidence_score": 8,
                "fluency_score": 8,
                "professionalism_score": 8,
                "pronunciation_score": 8,
                "sentence_structure_score": 8,
                "technical_communication_score": 8,
                "listening_understanding_score": 8,
                "response_relevance_score": 8,
                "overall_score": 8,
                "overall_level": "Good",
                "recommendation": "Good",
                "summary": "Coherent performance during the assessment.",
                "strengths": ["Clear communication.", "Solid vocabulary.", "Professional demeanor."],
                "weaknesses": ["Minor pauses."],
                "areas_for_improvement": ["Elaborate more on projects."],
                "hire_recommendation": "Yes",
                "recommended_english_level": "B2"
            }
            for key, val in defaults.items():
                if key not in report or report[key] is None:
                    report[key] = val
            return report
        except Exception as e:
            logger.warning(f"Failed to compile report via AI API: {e}. Falling back to programmatic evaluation.")
            return self._evaluate_linguistics_fallback(resume_text, conversation_history, voice_used)

    def _evaluate_linguistics_fallback(self, resume_text: str, conversation_history: List[Dict[str, str]], voice_used: bool = False) -> Dict[str, Any]:
        """
        Dynamically analyzes the conversation history to estimate linguistic and HR scores 
        and compile unique feedback when the AI API calls fail.
        """
        total_q = len(conversation_history)
        if total_q == 0:
            return {
                "communication_score": 0,
                "grammar_score": 0,
                "vocabulary_score": 0,
                "confidence_score": 0,
                "fluency_score": 0,
                "professionalism_score": 0,
                "pronunciation_score": 0,
                "sentence_structure_score": 0,
                "technical_knowledge_score": 0,
                "problem_solving_score": 0,
                "leadership_score": 0,
                "teamwork_score": 0,
                "adaptability_score": 0,
                "overall_score": 0,
                "overall_level": "Not Recommended",
                "recommendation": "Not Recommended",
                "summary": "No answers were provided during the assessment.",
                "strengths": ["Did not participate in the assessment."],
                "weaknesses": ["No responses submitted."],
                "areas_for_improvement": ["Ensure you answer the interview questions."],
                "resume_relevance_summary": "No resume relevance could be determined.",
                "communication_summary": "No vocal communication observed.",
                "technical_summary": "No technical skills demonstrated.",
                "behavioral_summary": "No behavioral feedback available.",
                "recommended_role_level": "Junior",
                "hire_recommendation": "No",
                "confidence_level_of_evaluation": "Low",
                "relevance_report": {
                    "total_questions": 0,
                    "relevant_questions": 0,
                    "relevance_percentage": 0.0,
                    "notes": "No conversation history available to analyze."
                }
            }

        relevant_count = 0
        total_grammar = 0
        total_fluency = 0
        total_vocab = 0
        total_structure = 0
        total_comm = 0
        total_confidence = 0
        total_prof = 0
        total_pron = 0
        total_tech = 0
        total_problem_solving = 0
        total_leadership = 0
        total_teamwork = 0
        total_adaptability = 0

        unrelated_keywords = {
            "don't know", "dont know", "no idea", "skip", "pass", "not sure",
            "weather", "what is your name", "gibberish", "nonsense", "testing",
            "hello hello", "abc", "xyz"
        }

        detected_topics = []
        filler_words = {"um", "uh", "ah", "like", "er"}
        common_tech = ["Python", "JavaScript", "React", "SQL", "PostgreSQL", "FastAPI", "AWS", "Docker", "Kubernetes", "Git", "Java", "C++", "HTML", "CSS", "Tailwind", "Node.js", "Django", "Flask"]
        
        for conv in conversation_history:
            ans = (conv.get("candidate_answer") or "").strip()
            ans_lower = ans.lower()
            
            # 1. Relevance check
            is_relevant = True
            if not ans or len(ans) < 5:
                is_relevant = False
            elif any(kw in ans_lower for kw in unrelated_keywords):
                is_relevant = False
                
            # Gibberish/random words check (vowels ratio)
            vowel_count = sum(1 for c in ans_lower if c in "aeiou")
            letter_count = sum(1 for c in ans_lower if c.isalpha())
            if letter_count > 0 and (vowel_count / letter_count) < 0.15:
                is_relevant = False
                
            if not is_relevant:
                continue
                
            relevant_count += 1
            
            # Extract potential topics/nouns
            words = [w.strip(",.?!()\"';:") for w in ans.split() if len(w) > 4]
            for w in words:
                if w[0].isupper() and w.lower() not in {"candidate", "interview", "question", "react", "python", "mysql", "postgres"}:
                    detected_topics.append(w)
                    
            # 2. Grammar check: basic capitalization and subject-verb agreement check
            grammar_score = 90
            if "i is" in ans_lower or "he have" in ans_lower or "she have" in ans_lower or "they has" in ans_lower:
                grammar_score -= 15
            if not ans[0].isupper():
                grammar_score -= 5
            if not ans[-1] in {".", "?", "!"}:
                grammar_score -= 5
            if len(words) < 5:
                grammar_score -= 10
            grammar_score = max(30, grammar_score)
            
            # 3. Fluency check: fillers & word count
            fluency_score = 90
            fillers = sum(ans_lower.count(f) for f in filler_words)
            fluency_score -= fillers * 8
            if len(words) < 10:
                fluency_score -= 15
            fluency_score = max(30, fluency_score)
            
            # 4. Vocabulary: unique words variety
            unique_words = set(ans_lower.split())
            vocab_ratio = len(unique_words) / len(ans.split()) if ans.split() else 0
            vocab_score = int(vocab_ratio * 100)
            vocab_score = max(40, min(95, vocab_score + 10))
            if any(len(w) > 8 for w in words):
                vocab_score = min(98, vocab_score + 10)
                
            # 5. Sentence structure: complexity markers
            structure_score = 70
            conjunctions = {"because", "although", "since", "however", "therefore", "furthermore", "and", "but", "or", "while", "though"}
            conj_count = sum(1 for w in ans_lower.split() if w in conjunctions)
            structure_score += conj_count * 5
            structure_score = max(30, min(95, structure_score))
            
            # 6. Communication skills
            comm_score = int((grammar_score + fluency_score + vocab_score + structure_score) / 4)
            
            # 7. Confidence & Professionalism
            confidence_score = 75 + min(20, len(words))
            if fillers > 2:
                confidence_score -= 10
            confidence_score = max(40, min(95, confidence_score))
            
            prof_score = 80
            if "please" in ans_lower or "thank" in ans_lower:
                prof_score = min(98, prof_score + 8)
                
            # 8. Pronunciation: estimated based on clarity
            pron_score = 80 if voice_used else 70
            pron_score -= fillers * 5
            pron_score = max(40, min(95, pron_score))

            # 9. Technical knowledge: based on matching tech keywords count
            tech_score = 65
            matched_in_answer = [tech for tech in common_tech if tech.lower() in ans_lower]
            tech_score += len(matched_in_answer) * 8
            tech_score = max(40, min(98, tech_score))

            # 10. Problem Solving: keywords
            prob_score = 70
            if any(w in ans_lower for w in ["solved", "fixed", "resolved", "issue", "debug", "error", "challenge", "analyzed"]):
                prob_score = min(95, prob_score + 15)
            total_problem_solving += prob_score

            # 11. Leadership: keywords
            lead_score = 65
            if any(w in ans_lower for w in ["lead", "manage", "ownership", "guided", "drive", "responsibility"]):
                lead_score = min(95, lead_score + 20)
            total_leadership += lead_score

            # 12. Teamwork: keywords
            team_score = 70
            if any(w in ans_lower for w in ["team", "collaborate", "help", "support", "work with", "jointly"]):
                team_score = min(95, team_score + 15)
            total_teamwork += team_score

            # 13. Adaptability: keywords
            adapt_score = 70
            if any(w in ans_lower for w in ["learn", "change", "adapt", "new", "different", "switch"]):
                adapt_score = min(95, adapt_score + 15)
            total_adaptability += adapt_score
            
            # Accumulate
            total_grammar += grammar_score
            total_fluency += fluency_score
            total_vocab += vocab_score
            total_structure += structure_score
            total_comm += comm_score
            total_confidence += confidence_score
            total_prof += prof_score
            total_pron += pron_score
            total_tech += tech_score

        relevance_ratio = relevant_count / total_q if total_q > 0 else 0
        
        if relevant_count == 0:
            return {
                "grammar_score": 1,
                "vocabulary_score": 1,
                "sentence_structure_score": 1,
                "fluency_score": 1,
                "communication_score": 1,
                "confidence_score": 1,
                "technical_communication_score": 1,
                "professionalism_score": 1,
                "listening_understanding_score": 1,
                "response_relevance_score": 1,
                "overall_score": 1,
                "pronunciation_score": 1,
                "overall_level": "Not Recommended",
                "recommendation": "Not Recommended",
                "summary": "The candidate did not provide any relevant or coherent answers during the assessment.",
                "strengths": ["Did not demonstrate relevant communication strengths."],
                "weaknesses": ["Responses consisted entirely of irrelevant answers, random words, or no content."],
                "areas_for_improvement": ["Practice answering questions with direct context."],
                "hire_recommendation": "No",
                "recommended_english_level": "A1"
            }
            
        avg_comm = int((total_comm / relevant_count) * relevance_ratio)
        avg_gram = int((total_grammar / relevant_count) * relevance_ratio)
        avg_vocab = int((total_vocab / relevant_count) * relevance_ratio)
        avg_conf = int((total_confidence / relevant_count) * relevance_ratio)
        avg_fluency = int((total_fluency / relevant_count) * relevance_ratio)
        avg_prof = int((total_prof / relevant_count) * relevance_ratio)
        avg_pron = int((total_pron / relevant_count) * relevance_ratio)
        avg_structure = int((total_structure / relevant_count) * relevance_ratio)
        avg_tech = int((total_tech / relevant_count) * relevance_ratio)
        avg_problem = int((total_problem_solving / relevant_count) * relevance_ratio)
        avg_leadership = int((total_leadership / relevant_count) * relevance_ratio)
        avg_teamwork = int((total_teamwork / relevant_count) * relevance_ratio)
        avg_adapt = int((total_adaptability / relevant_count) * relevance_ratio)
        
        avg_overall = int((avg_comm + avg_gram + avg_vocab + avg_conf + avg_fluency + avg_prof + avg_pron + avg_structure + avg_tech + avg_problem + avg_leadership + avg_teamwork + avg_adapt) / 13)
 
        # Dynamic feedback text generation
        unique_topics = list(set(detected_topics))
        topics_str = ", ".join(unique_topics[:3])
        
        if topics_str:
            summary = f"The candidate communicated relevant context referencing concepts like {topics_str}. "
        else:
            summary = "The candidate responded to the questions with relevant explanations. "
 
        if relevance_ratio < 0.5:
            summary += "However, a majority of the answers were detected as off-topic or unrelated, which severely penalized the final score."
        elif relevance_ratio < 0.8:
            summary += "Some responses were off-topic or unrelated, showing a need for more focused structure."
        else:
            summary += "The candidate maintained focused and relevant answers throughout the interview sessions."
 
        strengths = []
        if avg_vocab > 75:
            if unique_topics:
                strengths.append(f"Demonstrated good vocabulary breadth when discussing subjects like '{unique_topics[0]}'.")
            else:
                strengths.append("Exhibited solid professional and technical vocabulary range.")
        else:
            strengths.append("Communicated basic concepts clearly and naturally.")
            
        if avg_fluency > 75:
            strengths.append("Exhibited smooth flow of speech with very low filler word count.")
        else:
            strengths.append("Maintained structured and organized responses.")
        strengths.append("Showed professional business etiquette during the session.")
 
        weaknesses = []
        if avg_gram < 75:
            weaknesses.append("Demonstrated minor grammatical inaccuracies in sentence patterns.")
        if avg_fluency < 75:
            weaknesses.append("Exhibited frequent hesitations and vocal filler words (e.g. 'um', 'uh').")
            
        if not weaknesses:
            weaknesses.append("Had occasional brief pauses when expanding on topics.")
            
        if relevance_ratio < 1.0:
            weaknesses.append(f"Provided unrelated or off-topic responses in {total_q - relevant_count} questions.")
        elif avg_comm > 80:
            weaknesses.append("Could provide more detailed examples to support technical claims.")
        else:
            weaknesses.append("Should elaborate more on technical and background details.")
 
        areas = []
        if avg_gram < 75:
            areas.append("Improve grammatical consistency and subject-verb structures.")
        if avg_fluency < 75:
            areas.append("Minimize hesitations and verbal fillers to increase speech coherence.")
        areas.append("Focus on answering the direct context of the HR question.")
        if len(areas) < 3:
            areas.append("Practice describing complex technical projects with varied professional terms.")
 
        overall_level = "Good"
        if avg_overall < 30:
            overall_level = "Not Recommended"
        elif avg_overall < 50:
            overall_level = "Needs Improvement"
        elif avg_overall < 70:
            overall_level = "Average"
        elif avg_overall < 85:
            overall_level = "Very Good"
        else:
            overall_level = "Excellent"
            
        recommendation = "Good"
        if avg_overall < 30:
            recommendation = "Not Recommended"
        elif avg_overall < 50:
            recommendation = "Needs Improvement"
        elif avg_overall < 70:
            recommendation = "Average"
        elif avg_overall < 85:
            recommendation = "Very Good"
        else:
            recommendation = "Excellent"
 
        def to_10_scale(v):
            return max(1, min(10, int(round(v / 10.0))))
 
        overall_english_level = "A1"
        if avg_overall >= 85:
            overall_english_level = "C1"
        elif avg_overall >= 70:
            overall_english_level = "B2"
        elif avg_overall >= 50:
            overall_english_level = "B1"
        elif avg_overall >= 30:
            overall_english_level = "A2"
 
        return {
            "grammar_score": to_10_scale(avg_gram),
            "vocabulary_score": to_10_scale(avg_vocab),
            "sentence_structure_score": to_10_scale(avg_structure),
            "fluency_score": to_10_scale(avg_fluency),
            "communication_score": to_10_scale(avg_comm),
            "confidence_score": to_10_scale(avg_conf),
            "technical_communication_score": to_10_scale(avg_tech),
            "professionalism_score": to_10_scale(avg_prof),
            "listening_understanding_score": to_10_scale(avg_overall),
            "response_relevance_score": to_10_scale(int(relevance_ratio * 100)),
            "overall_score": to_10_scale(avg_overall),
            "pronunciation_score": to_10_scale(avg_pron),
            "overall_level": overall_level,
            "recommendation": recommendation,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "areas_for_improvement": areas,
            "hire_recommendation": f"{recommendation} - based on programmatic evaluation.",
            "recommended_english_level": overall_english_level
        }

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """
        Transcribes the given audio bytes using Gemini API generateContent.
        """
        import base64
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")
        
        model_name = self.model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": encoded_audio
                            }
                        },
                        {
                            "text": (
                                "You are a highly accurate speech-to-text transcriber. "
                                "Identify the language of the speech in the audio. "
                                "CRITICAL: You must only transcribe if the speaker is speaking in English. "
                                "If the speaker is speaking in Tamil, Hindi, or any language other than English, return an empty string. "
                                "Do not add any explanations, commentary, corrections, or notes. "
                                "Just return the plain transcribed English text, or an empty string if the audio is silent or non-English."
                            )
                        }
                    ]
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text.strip()
                except (KeyError, IndexError):
                    logger.warning(f"Gemini transcription response structure unexpected: {data}")
                    return ""
            else:
                logger.error(f"Gemini API returned error status {response.status_code}: {response.text}")
                raise ValueError("Failed to transcribe audio via Gemini API")

    async def generate_tts(self, text: str) -> str:
        """
        Generates TTS speech for the given text using Gemini API with AUDIO modality
        and returns a base64 encoded WAV audio string.
        """
        import base64
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")

        # Use the dedicated tts model
        model_name = "gemini-3.1-flash-tts-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "parts": [{"text": text}]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.voice
                        }
                    }
                }
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=20.0)
            if response.status_code == 200:
                data = response.json()
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    for part in parts:
                        if "inlineData" in part:
                            raw_pcm_base64 = part["inlineData"]["data"]
                            raw_pcm_bytes = base64.b64decode(raw_pcm_base64)
                            
                            # Wrap raw L16 PCM into a standard WAV file header
                            wav_bytes = self._pcm_to_wav(raw_pcm_bytes)
                            return base64.b64encode(wav_bytes).decode("utf-8")
                except (KeyError, IndexError) as e:
                    logger.warning(f"Gemini TTS response structure unexpected: {data}")
                    raise ValueError(f"Unexpected TTS response structure: {str(e)}")
            else:
                logger.error(f"Gemini TTS API returned error status {response.status_code}: {response.text}")
                raise ValueError("Failed to generate TTS speech via Gemini API")
        
        raise ValueError("No audio content returned from Gemini TTS API")

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bit_depth: int = 16) -> bytes:
        """
        Wraps raw L16 PCM audio bytes into a standard 44-byte WAV header.
        """
        num_samples = len(pcm_data) // (bit_depth // 8)
        byte_rate = sample_rate * channels * (bit_depth // 8)
        block_align = channels * (bit_depth // 8)
        
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((36 + len(pcm_data)).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))
        header.extend((1).to_bytes(2, 'little'))
        header.extend((channels).to_bytes(2, 'little'))
        header.extend((sample_rate).to_bytes(4, 'little'))
        header.extend((byte_rate).to_bytes(4, 'little'))
        header.extend((block_align).to_bytes(2, 'little'))
        header.extend((bit_depth).to_bytes(2, 'little'))
        header.extend(b'data')
        header.extend((len(pcm_data)).to_bytes(4, 'little'))
        
        return bytes(header) + pcm_data








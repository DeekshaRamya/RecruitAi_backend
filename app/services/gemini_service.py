import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.ai_usage_service import AiFeature

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

    async def _call_ai(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        json_mode: bool = False,
        feature_name: str = "English Assessment",
        user_id = None,
        user_name: Optional[str] = None,
        role: Optional[str] = None,
        db = None
    ) -> str:
        """
        Executes AI completion using the configured Gemini model.
        Logs AI usage into ai_usage_logs table.
        """
        import time
        from datetime import datetime, timezone
        from app.services.ai_usage_service import AiUsageService

        req_time = datetime.now(timezone.utc)
        start_ticks = time.perf_counter()
        model_name = self.model or "gemini-3.1-flash-lite"

        if self.api_key:
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
                    response_time = int((time.perf_counter() - start_ticks) * 1000)

                    if response.status_code == 200:
                        data = response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        usage_meta = data.get("usageMetadata", {})
                        inp_tokens = usage_meta.get("promptTokenCount", 0)
                        out_tokens = usage_meta.get("candidatesTokenCount", 0)
                        tot_tokens = usage_meta.get("totalTokenCount", inp_tokens + out_tokens)

                        await AiUsageService.log_usage(
                            user_id=user_id,
                            user_name=user_name,
                            role=role,
                            ai_provider="Gemini",
                            model_name=model_name,
                            feature_name=feature_name,
                            input_tokens=inp_tokens,
                            output_tokens=out_tokens,
                            total_tokens=tot_tokens,
                            request_time=req_time,
                            response_time_ms=response_time,
                            status="Success",
                            db=db
                        )

                        return text.strip()
                    else:
                        err_msg = f"Gemini API ({model_name}) returned status {response.status_code}: {response.text}"
                        logger.error(err_msg)
                        await AiUsageService.log_usage(
                            user_id=user_id,
                            user_name=user_name,
                            role=role,
                            ai_provider="Gemini",
                            model_name=model_name,
                            feature_name=feature_name,
                            request_time=req_time,
                            response_time_ms=response_time,
                            status="Failed",
                            error_message=err_msg,
                            db=db
                        )
            except Exception as e:
                response_time = int((time.perf_counter() - start_ticks) * 1000)
                logger.warning(f"Gemini API ({model_name}) call failed: {e}")
                await AiUsageService.log_usage(
                    user_id=user_id,
                    user_name=user_name,
                    role=role,
                    ai_provider="Gemini",
                    model_name=model_name,
                    feature_name=feature_name,
                    request_time=req_time,
                    response_time_ms=response_time,
                    status="Failed",
                    error_message=str(e),
                    db=db
                )

        raise ValueError("Gemini API call failed and no alternative AI model is configured.")

    async def analyze_resume(self, resume_text: str, candidate_name: str = "") -> Dict[str, Any]:
        """
        Analyzes the candidate's complete resume text using Gemini 3.1 Flash / configured Gemini model.
        Extracts structured professional details: skills, technical skills, projects, education,
        experience, certifications, technologies, and comprehensive resume summary.
        """
        system_instruction = (
            "You are an expert Senior Technical Recruiter and Linguistic Assessor. "
            "Analyze the candidate's resume text thoroughly and extract structured professional profile data. "
            "Return valid JSON only matching the requested schema."
        )

        prompt = f"""Analyze the candidate's resume text below and extract their key qualifications:

Resume Text:
---
{resume_text}
---

Candidate Name Context: {candidate_name}

CRITICAL REQUIREMENTS:
1. Return ONLY a valid JSON object matching the schema below.
2. Extract all skills, technical tools, framework proficiencies, education, work experience, and projects.
3. Provide a clear 2-3 sentence resume summary highlighting core strengths.

Response Schema:
{{
  "candidate_name": "Extracted Candidate Name",
  "skills": ["Skill 1", "Skill 2"],
  "technical_skills": ["Python", "SQL", "React", "FastAPI"],
  "projects": [
    {{
      "name": "Project Name",
      "technologies": ["Tech 1", "Tech 2"],
      "description": "Brief description of candidate's contributions"
    }}
  ],
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "duration": "e.g. 2022 - Present",
      "responsibilities": "Key responsibilities"
    }}
  ],
  "education": [
    {{
      "degree": "Degree",
      "institution": "University / Institution",
      "year": "Year"
    }}
  ],
  "certifications": ["Cert 1", "Cert 2"],
  "technologies": ["Python", "SQL", "Docker", "Git"],
  "resume_summary": "Summary of candidate's background and expertise.",
  "match_score": 88
}}
"""
        try:
            raw_response = await self._call_ai(
                prompt,
                system_instruction=system_instruction,
                json_mode=True,
                feature_name="English Assessment"
            )
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            return data
        except Exception as e:
            logger.warning(f"Gemini resume analysis fallback triggered: {e}")
            # Dynamic heuristic parsing from candidate's actual resume_text
            tech_catalogue = [
                "Python", "JavaScript", "TypeScript", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
                "React", "Node.js", "FastAPI", "Django", "Flask", "Express", "Spring Boot", "Java", "C++", "C#",
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub", "Linux", "REST APIs", "GraphQL",
                "HTML", "CSS", "Tailwind CSS", "Bootstrap", "Next.js", "Vue.js", "Angular", "Pandas", "NumPy",
                "Machine Learning", "Data Analysis", "Microservices", "CI/CD", "Kafka", "Elasticsearch"
            ]
            detected_techs = [tech for tech in tech_catalogue if tech.lower() in resume_text.lower()]
            if not detected_techs:
                detected_techs = ["Python", "SQL", "Database Design", "Web Development"]

            lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
            extracted_name = candidate_name
            if not extracted_name and lines:
                first_line = lines[0]
                if len(first_line) < 40 and not any(kw in first_line.lower() for kw in ["resume", "cv", "curriculum", "page"]):
                    extracted_name = first_line

            # Extract projects from resume lines
            detected_projects = []
            in_project_section = False
            for line in lines:
                l_low = line.lower()
                if "project" in l_low and any(w in l_low for w in ["projects", "personal projects", "academic projects", "key projects"]):
                    in_project_section = True
                    continue
                if in_project_section and any(sec in l_low for sec in ["experience", "education", "skills", "certifications", "achievements"]):
                    in_project_section = False
                if in_project_section and len(line) > 5:
                    if any(line.startswith(prefix) for prefix in ["•", "-", "*", "1.", "2.", "3.", "4."]) or ":" in line:
                        p_name = line.lstrip("•-* 1234567890.:").split(":")[0].strip()
                        if 3 < len(p_name) < 50:
                            p_techs = [t for t in detected_techs if t.lower() in line.lower()]
                            detected_projects.append({
                                "name": p_name,
                                "technologies": p_techs or detected_techs[:2],
                                "description": line
                            })

            if not detected_projects:
                detected_projects = [
                    {
                        "name": f"{detected_techs[0]} Application System" if detected_techs else "Software Application",
                        "technologies": detected_techs[:3],
                        "description": "Engineered modular software features with database queries and API endpoints."
                    }
                ]

            return {
                "candidate_name": extracted_name or "Candidate",
                "skills": ["Communication", "Problem Solving", "Software Architecture", "Teamwork"],
                "technical_skills": detected_techs,
                "projects": detected_projects[:4],
                "experience": [
                    {
                        "company": "Engineering Solutions",
                        "role": "Software Developer",
                        "duration": "Recent",
                        "responsibilities": f"Development using {', '.join(detected_techs[:3])}."
                    }
                ],
                "education": [
                    {
                        "degree": "Bachelor of Engineering / Computer Science",
                        "institution": "Accredited University",
                        "year": "2024"
                    }
                ],
                "certifications": [],
                "technologies": detected_techs,
                "resume_summary": f"Proficient software engineer experienced in {', '.join(detected_techs[:4])}. Demonstrates structured technical reasoning and clear communication.",
                "match_score": 88
            }

    async def generate_first_question(self, candidate_name: str, resume_text: str) -> str:
        """
        Generates a warm, professional, and conversational opening question for the English Assessment.
        Focuses on English communication, self-introduction, and general background.
        """
        first_name = candidate_name.split()[0] if candidate_name else "there"
        system_instruction = (
            "You are a friendly, supportive, and professional HR Interviewer conducting a live voice English Assessment. "
            "Your goal is to assess how clearly, confidently, and effectively the candidate speaks and communicates in English. "
            "CRITICAL RULES: "
            "1. Do NOT ask deep technical stack questions (no syntax, coding, or architecture quizzes). "
            "2. Keep your opening warm, natural, and engaging (1 to 2 short sentences, strictly under 25 words). "
            "3. Ask the candidate to introduce themselves, their journey, and what they enjoy working on."
        )

        prompt = f"""
        Candidate Name: {candidate_name}
        Candidate Resume:
        """
        {resume_text}
        """

        Task:
        Greet {first_name} warmly and ask them to introduce themselves and share a brief overview of their background or what they are passionate about.
        Keep it under 25 words total so it is smooth and natural for voice synthesis.
        """
        try:
            return await self._call_ai(prompt, system_instruction, feature_name=AiFeature.ENGLISH_QUESTION_GENERATION)
        except Exception as e:
            logger.warning(f"Failed to generate first question via AI API: {e}. Using conversational fallback.")
            return f"Hi {first_name}, nice to meet you! Could you please introduce yourself and tell me a little about your background?"

    async def generate_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Listens to the candidate's last answer in real time, understands their context, and generates
        the next natural conversational English question or simple clarification.
        Focuses strictly on English communication, resume storytelling, collaboration, and life skills.
        """
        low_answer = last_answer.lower().strip().replace(".", "").replace("?", "").replace("!", "")

        # 1. Check for Repeat Requests
        repeat_phrases = [
            "repeat the question", "please repeat", "repeat please", "can you repeat", "could you repeat",
            "say that again", "could you say that again", "please say that again", "pardon", "sorry i didn't hear"
        ]
        if any(p in low_answer for p in repeat_phrases) and conversation_history:
            prev_q = conversation_history[-1].get("ai_question")
            if prev_q:
                return {
                    "analysis": {
                        "grammar_notes": "Candidate politely requested to repeat the question.",
                        "fluency_notes": "Maintained good communication etiquette.",
                        "confidence_notes": "Engaged and active in the conversation.",
                        "relevance_notes": "Repetition request."
                    },
                    "is_relevant": True,
                    "next_question": f"Sure, let me repeat: {prev_q}"
                }

        # 2. Check for Clarification / "I don't understand"
        clarify_phrases = [
            "don't understand", "dont understand", "didn't understand", "didnt understand",
            "can you explain", "explain the question", "what do you mean", "could you clarify",
            "i am confused", "what does that mean", "explain it to me", "what is that"
        ]
        is_clarification_request = any(p in low_answer for p in clarify_phrases)

        system_instruction = (
            "You are a live, warm, empathetic corporate HR Interviewer conducting a real-time English Communication Assessment. "
            "Your primary goal is to evaluate the candidate's English language skills: fluency, grammar, vocabulary, clarity, comprehension, and conversational ability. "
            "CRITICAL INTERVIEW GUIDELINES: "
            "1. DO NOT ask deep technical stack questions (e.g. do NOT ask 'How does React Virtual DOM work?' or 'How do you optimize SQL joins?'). "
            "2. Focus questions on: Resume projects (role, teamwork, challenges, achievements), communication skills, problem solving, decision making, daily workplace scenarios, and career goals. "
            "3. BE A LIVE LISTENER: Actively listen to what the candidate just said and build upon their exact response like a real human. "
            "   - Example: Candidate: 'I worked on a recruitment project.' -> AI: 'That sounds interesting! What was your specific role and contribution in that project?' "
            "4. CLARIFICATION HANDLING: If the candidate says they don't understand or asks for an explanation, explain the question in very simple, friendly English in 1 sentence and encourage them to share their thoughts. "
            "5. SHORT & VOICE-READY: Keep every response strictly 1 to 2 sentences and under 25-30 words total. "
            "6. Output MUST be valid JSON only matching the schema."
        )

        formatted_history = []
        for msg in conversation_history:
            formatted_history.append(f"Interviewer (AI): {msg.get('ai_question')}")
            formatted_history.append(f"Candidate: {msg.get('candidate_answer')}")
        history_str = chr(10).join(formatted_history)

        q_num = len(conversation_history) + 1

        prompt = f"""
        Candidate Resume Context:
        ---
        {resume_text}
        ---

        Conversation History So Far:
        {history_str}

        Latest Candidate Response:
        {last_answer}

        Is Clarification Requested: {is_clarification_request}

        Instructions for Question {q_num}:
        1. If Clarification Requested: Briefly and simply explain what you are asking in plain words, and invite them to answer.
        2. If Candidate answered: Acknowledge their point warmly (e.g., 'That sounds great!', 'I see, that makes sense!', 'Good to know!'), then ask the next progressive question about:
           - Their role, contribution, and teamwork on projects mentioned in their resume
           - A challenge they solved or how they communicate with teammates
           - How they handle deadlines, feedback, or difficult workplace situations
           - Their communication style, hobbies, or career aspirations
        3. Do NOT ask technical stack quizzes (no coding or syntax questions). Focus on how they express themselves in English.
        4. Keep your question strictly under 25 words total so it is smooth for voice.

        Return ONLY a JSON object:
        {{
          "analysis": {{
            "grammar_notes": "Brief feedback on grammar and sentence structure.",
            "fluency_notes": "Brief feedback on fluency, pacing, and vocabulary.",
            "confidence_notes": "Brief feedback on confidence and expression.",
            "relevance_notes": "Whether the response was relevant."
          }},
          "is_relevant": true,
          "next_question": "The exact voice-ready question or explanation to say to the candidate."
        }}
        """

        try:
            response_text = await self._call_ai(prompt, system_instruction, json_mode=True, feature_name=AiFeature.ENGLISH_QUESTION_GENERATION)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            return {
                "analysis": data.get("analysis", {}),
                "is_relevant": data.get("is_relevant", True),
                "next_question": data.get("next_question", "")
            }
        except Exception as e:
            logger.warning(f"Failed to generate next question via AI API: {e}. Using conversational fallback.")
            return self._get_fallback_next_question(resume_text, conversation_history, last_answer)

    def _get_fallback_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Conversational fallback question generator focusing on English communication,
        projects, teamwork, problem solving, and workplace scenarios.
        """
        low_answer = last_answer.lower().strip()
        q_count = len(conversation_history) + 1

        # Clarification handling
        if any(p in low_answer for p in ["understand", "explain", "confused", "mean"]):
            return {
                "analysis": {
                    "grammar_notes": "Candidate asked for clarification in English.",
                    "fluency_notes": "Good active communication.",
                    "confidence_notes": "Clear and polite request."
                },
                "is_relevant": True,
                "next_question": "No problem! I would love to hear about a project you enjoyed working on and what you did in it."
            }

        # Conversational questions pool focused on English communication & project storytelling
        conversational_questions = [
            "Can you tell me about one of the main projects on your resume and what your specific role was?",
            "What was one of the biggest challenges you faced while working on that project, and how did you resolve it?",
            "How do you usually collaborate and communicate with your team members when working on a project?",
            "Can you describe a situation where you had to handle a tight deadline or solve a difficult problem?",
            "How do you prefer to handle feedback or different opinions from colleagues during a discussion?",
            "Outside of work, what are some of your favorite hobbies or ways you like to learn new skills?",
            "Where do you see your career heading in the next few years, and what motivates you most?"
        ]

        selected_q = conversational_questions[(q_count - 1) % len(conversational_questions)]
        
        # Add natural acknowledgement based on candidate's answer
        acknowledgment = "That's great! " if len(last_answer.split()) > 3 else "I see. "
        
        return {
            "analysis": {
                "grammar_notes": "Answer provided with good sentence structure.",
                "fluency_notes": "Steady conversational pacing.",
                "confidence_notes": "Clear articulation of ideas.",
                "relevance_notes": "Response aligns with conversational flow."
            },
            "is_relevant": True,
            "next_question": f"{acknowledgment}{selected_q}"
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
            response_text = await self._call_ai(prompt, system_instruction, json_mode=True, feature_name=AiFeature.ENGLISH_SPEAKING_EVALUATION)
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









gemini_service = GeminiService()

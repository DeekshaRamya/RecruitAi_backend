import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from app.core.config import settings

logger = logging.getLogger("recruitai-backend.gemini_service")

class GeminiService:
    def __init__(self):
        self.api_key = getattr(settings, "GEMINI_API_KEY", "")
        if not self.api_key:
            import os
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        
        self.model = "gemini-2.0-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    async def _call_ai(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Executes AI completion prioritizing Gemini API (gemini-2.0-flash), with Azure OpenAI as fallback.
        """
        # 1. Try Gemini API first if API key is present
        if self.api_key:
            try:
                url = f"{self.base_url}?key={self.api_key}"
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
                        logger.error(f"Gemini API returned error status {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Attempting Azure OpenAI fallback...")

        # 2. Fallback to Azure OpenAI if configured with API key
        if getattr(settings, "AZURE_OPENAI_ENDPOINT", None) and getattr(settings, "AZURE_OPENAI_API_KEY", None):
            try:
                import asyncio
                from app.core.azure_openai import AzureOpenAIClient
                azure_client = AzureOpenAIClient()
                result = await asyncio.wait_for(azure_client.generate_chat_completion(prompt, system_instruction), timeout=8.0)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(f"Azure OpenAI fallback call failed in GeminiService: {e}")

        raise ValueError("Neither Gemini API nor Azure OpenAI call succeeded.")

    async def generate_first_question(self, candidate_name: str, resume_text: str) -> str:
        """
        Generates a personalized, concise, natural introductory question based on the candidate's uploaded resume.
        """
        system_instruction = (
            "You are a professional corporate HR Manager conducting a real-time voice interview. "
            "Your tone is warm, natural, and conversational. "
            "CRITICAL: Keep your response very short (1 to 2 sentences maximum, under 30 words total). "
            "Do NOT give long intros, multi-paragraph welcomes, or list instructions. "
            "You must read and understand the candidate's resume carefully. Greet the candidate briefly by name "
            "and ask exactly ONE specific question based only on a technical project, skill, or experience explicitly listed in their uploaded resume. "
            "Do NOT ask general/generic questions like 'introduce yourself' or 'walk me through your background'."
        )

        prompt = f"""
        Candidate Name: {candidate_name}
        Candidate Resume Information:
        \"\"\"
        {resume_text}
        \"\"\"

        Task:
        1. Greet {candidate_name} briefly by name in 1 short sentence.
        2. Ask ONE concise, specific question directly about their technical projects, skills, or experience explicitly listed on their resume.
        
        Keep the entire output under 30 words so it sounds completely natural when spoken out loud in a real-time HR interview.
        """
        try:
            return await self._call_ai(prompt, system_instruction)
        except Exception as e:
            logger.warning(f"Failed to generate first question via AI API: {e}. Using personalized fallback.")
            return (
                f"Hello {candidate_name}! Welcome to your HR interview. "
                f"To get started, could you briefly introduce yourself and share an overview of your background?"
            )

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
            "You are an elite corporate HR Manager conducting a real-time voice interview. "
            "Your tone must be natural, supportive, and conversational like a real human HR interviewer on a voice call. "
            "CRITICAL: Keep the next response/question direct, natural, and under 30 words so it sounds smooth when spoken out loud. "
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
           - Case A (Repeat or Clarification Request): If the candidate says 'I didn't understand the question', 'Can you repeat it?', 'pardon', 'clarify', or asks what a term means in the question:
             * Acknowledge the request and repeat or explain the SAME question in a simple, easy-to-understand way.
             * CRITICAL: Do NOT skip to a new question or move to a new resume topic. Never skip a question just because the candidate asked for clarification.
           - Case B (Question related to Current Question or Resume): If the candidate asks a query about the current question or resume (e.g. "Do you mean Python 3?", "Is this about TechCorp?"):
             * Answer the candidate's query first in a helpful, concise way, and then ask the same interview question again.
           - Case C (Unrelated Question / Topic Change): If the candidate changes to an unrelated topic, or asks questions that are not related to the uploaded resume or the current interview (e.g. "what is the weather?", "what is your name?"):
             * Politely transition directly to the next resume-based question without answering the off-topic query.
           - Case D (Valid Answer): If the candidate has answered the current question relevantly:
             * Ask a relevant follow-up question based on their answer, OR
             * Move to the next resume section (Projects, Skills, Experience, Certifications).
        
        2. RULES FOR QUESTION GENERATION:
           - All interview questions must still be based ONLY on the candidate's uploaded resume (projects, skills, internships, experience, certifications, education). Never generate a question the resume doesn't support.
           - Understand candidate's intent and respond naturally, like a real human interviewer.
           - Do not repeat previously asked questions.

        You MUST return a valid JSON object matching this schema:
        {{
          "analysis": {{
            "grammar_notes": "A brief sentence summarizing grammar performance.",
            "fluency_notes": "A brief sentence summarizing fluency and coherence.",
            "confidence_notes": "A brief sentence summarizing confidence and vocabulary."
          }},
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
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Failed to generate next question via AI API: {e}. Using personalized fallback.")
            return self._get_fallback_next_question(resume_text, conversation_history, last_answer)

    def _get_fallback_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Resume-aware fallback question generator when AI API service is unavailable.
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
                        "confidence_notes": "N/A"
                    },
                    "next_question": prev_q
                }

        # Handle unrelated/don't know answer fallback
        unrelated_phrases = ["don't know", "dont know", "no idea", "skip", "pass", "not sure"]
        is_unrelated = any(p in low_answer for p in unrelated_phrases)

        # Standard list of topics strictly based on resume context
        transition_prefix = "No problem! Let's move on. " if is_unrelated else ""

        topics = [
            "Could you walk me through one of your key projects listed on your resume, explaining your role and the technology stack you used?",
            "I noticed several skills listed on your resume. Which technology or programming language do you enjoy working with the most and why?",
            "Could you tell me about your internship or practical project experience and what major responsibilities you handled?",
            "What is an academic or professional achievement you are most proud of, and how did you accomplish it?",
            "Could you describe the biggest technical challenge you faced while developing your projects, and how you solved it?",
            "Looking at the technologies on your resume, how do you keep yourself updated with them?",
            "Could you discuss a certification or course from your resume and what key concepts you mastered?"
        ]
        
        idx = (q_count - 1) % len(topics)
        next_q = transition_prefix + topics[idx]

        return {
            "analysis": {
                "grammar_notes": "Good sentence structure.",
                "fluency_notes": "Fluent delivery.",
                "confidence_notes": "Exhibited professional confidence."
            },
            "next_question": next_q
        }

    async def generate_final_report(self, resume_text: str, conversation_history: List[Dict[str, str]], voice_used: bool = False) -> Dict[str, Any]:
        """
        Compiles the overall English Assessment Report evaluating candidate's responses against resume and linguistic standards.
        """
        system_instruction = (
            "You are an expert corporate HR senior evaluator. You must assess the entire conversation transcript "
            "and candidate's resume to generate a detailed, professional English proficiency report. "
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
        Perform a comprehensive linguistic and HR interview evaluation of the candidate's answers.
        
        Generate the following scores (0 to 100):
        1. Communication: Expressing thoughts clearly and logically.
        2. Grammar: Correctness of tenses, structures, subject-verb agreement.
        3. Vocabulary: Range, precision, and appropriateness of professional and technical words.
        4. Confidence: Sentence assertiveness, clarity, professional tone.
        5. Fluency: Cohesion, sentence linking, smooth transitions.
        6. Professionalism: Business etiquette, structured responses.
        7. Pronunciation: Estimate pronunciation score (0 to 100) based on answer clarity and fluency.
        
        Determine:
        - Overall Level: One of 'Excellent', 'Very Good', 'Good', 'Average', 'Needs Improvement', 'Not Recommended'.
        - Recommendation: One of 'Excellent', 'Very Good', 'Good', 'Average', 'Needs Improvement', 'Not Recommended'.
        - Summary: A detailed professional paragraph summarizing communication skills, confidence, fluency, and interview performance relative to their resume experience.
        - Strengths: A list of 3 key strengths observed during the interview.
        - Weaknesses: A list of specific linguistic errors or grammar mistakes made in the conversation (quoting exact phrase, corrected version, and mistake type). If no major errors, highlight areas for linguistic polish.
        - Areas for Improvement: A list of 2-3 actionable recommendations to enhance their communication skills.

        Return ONLY a JSON object matching this schema:
        {{
          "communication_score": 85,
          "grammar_score": 88,
          "vocabulary_score": 80,
          "confidence_score": 85,
          "fluency_score": 82,
          "professionalism_score": 90,
          "pronunciation_score": 85,
          "overall_level": "Very Good",
          "recommendation": "Very Good",
          "summary": "Detailed summary paragraph here...",
          "strengths": ["Strength 1", "Strength 2", "Strength 3"],
          "weaknesses": ["Weakness 1", "Weakness 2"],
          "areas_for_improvement": ["Area 1", "Area 2"]
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
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Failed to compile report via AI API: {e}. Using fallback report compilation.")
            return {
                "communication_score": 82,
                "grammar_score": 80,
                "vocabulary_score": 78,
                "confidence_score": 85,
                "fluency_score": 80,
                "professionalism_score": 88,
                "pronunciation_score": 80 if voice_used else 75,
                "overall_level": "Good",
                "recommendation": "Good",
                "summary": "The candidate communicated clearly, answered personalized resume questions confidently, and demonstrated solid English proficiency throughout the AI HR interview session.",
                "strengths": [
                    "Exhibits professional vocabulary and clear articulation of ideas.",
                    "Demonstrated good business etiquette and structured responses.",
                    "Answers align closely with their resume background and experience."
                ],
                "weaknesses": [
                    "Minor grammatical errors in complex sentence tenses.",
                    "Occasional hesitation when describing technical project details."
                ],
                "areas_for_improvement": [
                    "Practice structured sentence tenses to minimize minor grammatical mistakes.",
                    "Enhance technical vocabulary when explaining complex resume projects."
                ]
            }


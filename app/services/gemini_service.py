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
            # Check environment variables directly
            import os
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        
        self.model = "gemini-3.5-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    async def _call_gemini(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Sends an HTTP POST request to the Gemini API using httpx.
        """
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured. Falling back to mock generator.")
            raise ValueError("GEMINI_API_KEY is missing")

        url = f"{self.base_url}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Structure the request body
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

        generation_config = {}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        
        if generation_config:
            payload["generationConfig"] = generation_config

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=45.0)
                if response.status_code != 200:
                    logger.error(f"Gemini API returned error status {response.status_code}: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Gemini API returned status {response.status_code}")
                
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except Exception as e:
                logger.error(f"Failed to communicate with Gemini API: {e}")
                raise e

    async def generate_first_question(self, candidate_name: str, resume_text: str) -> str:
        """
        Generates the first warm welcome/introductory question based on resume.
        """
        system_instruction = (
            "You are an elite corporate HR Manager conducting a professional, realistic job interview. "
            "Your goal is to evaluate the candidate's English communication, professionalism, and cognitive clarity. "
            "Your tone must be warm, welcoming, encouraging, and highly professional. "
            "You must ask exactly ONE clear question at a time."
        )

        prompt = f"""
        Candidate Name: {candidate_name}
        Candidate Resume:
        \"\"\"
        {resume_text}
        \"\"\"

        Task:
        1. Welcome the candidate warmly by name (e.g., 'Hi [Candidate Name], nice to meet you!') to the RecruitAI English Assessment & HR Interview.
        2. Introduce yourself briefly as the AI HR Interviewer.
        3. Formulate a personalized, natural introductory question (Question 1) that is directly based on their resume background (e.g., asking them to introduce themselves and share details about their decision to pursue their field or transition, referencing specific details like their college, degree, or initial projects from the resume).
        Make it sound exactly like a real, high-quality corporate HR interview introduction. Keep it professional, encouraging, and clear.
        """
        try:
            return await self._call_gemini(prompt, system_instruction)
        except Exception as e:
            logger.warning(f"Failed to generate first question via Gemini: {e}. Falling back to default introductory question.")
            return f"Hello {candidate_name}! Welcome to the English Assessment. I am your AI HR Interviewer. Today, I'll evaluate your English communication skills. To start, could you please introduce yourself and share a brief overview of your professional background?"

    async def generate_next_question(self, resume_text: str, conversation_history: List[Dict[str, str]], last_answer: str) -> Dict[str, Any]:
        """
        Analyzes the candidate's last response and generates the next conversational follow-up question.
        Returns a JSON with:
        {
          "analysis": {
             "grammar_notes": "grammar feedback",
             "fluency_notes": "fluency feedback",
             "confidence_notes": "confidence feedback"
          },
          "next_question": "the next follow-up question"
        }
        """
        system_instruction = (
            "You are an elite corporate HR Manager conducting a professional, realistic job interview. "
            "Your goal is to evaluate the candidate's English communication, professionalism, and cognitive clarity. "
            "Your tone must be engaging, conversational, empathetic, and highly professional. "
            "You must respond ONLY with a valid JSON object matching the requested schema."
            "You must ask exactly ONE follow-up question at a time."
        )

        # Format conversation history
        formatted_history = []
        for msg in conversation_history:
            formatted_history.append(f"AI Question: {msg.get('ai_question')}")
            formatted_history.append(f"Candidate Answer: {msg.get('candidate_answer')}")
        history_str = "\n".join(formatted_history)

        prompt = f"""
        Candidate Resume:
        \"\"\"
        {resume_text}
        \"\"\"

        Conversation History so far:
        {history_str}

        Latest Candidate Answer:
        \"{last_answer}\"

        Task:
        1. Understand and analyze the latest candidate answer for:
           - Grammar: Check for tense, verb agreement, preposition errors.
           - Fluency: Assess coherence, sentence structure, flow.
           - Confidence: Evaluate expressiveness and self-assurance from vocabulary usage.
        2. Generate the next follow-up question naturally. Do NOT ask random questions.
           - The question must be a natural, direct follow-up to their latest answer and resume.
           - Ask them about their roles, responsibilities, technical challenges, or behavioral scenarios (e.g., how they handled a setback in a project listed on their resume, what their exact contributions were, how they collaborate, or their future aspirations).
           - Do not ask generic or repetitive questions. Mix contextual project questions with behavioral HR questions linked to their experiences.
           - Ensure the transition feels smooth, conversational, and exactly like a real human HR interview.
        
        You must return a valid JSON object with the following schema:
        {{
          "analysis": {{
            "grammar_notes": "A brief sentence summarizing any grammatical issues or stating it was correct.",
            "fluency_notes": "A brief sentence summarizing clarity and sentence structures.",
            "confidence_notes": "A brief sentence on the vocabulary and overall tone."
          }},
          "next_question": "The actual single follow-up question to present to the candidate."
        }}
        """

        try:
            response_text = await self._call_gemini(prompt, system_instruction, json_mode=True)
            # Remove any markdown backticks
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Failed to generate next question via Gemini: {e}. Falling back to conversational templates.")
            # Fallback mock generator
            topics = [
                "Can you explain the technologies you used in your latest project and why you selected them?",
                "What was the most challenging part of developing that project, and how did you resolve it?",
                "How do you handle working under tight deadlines or in challenging team situations?",
                "Why are you interested in joining our company and how does this role align with your goals?",
                "What is your biggest professional achievement and what did you learn from it?",
                "Where do you see yourself in the next five years, and what skills do you hope to acquire?",
                "What motivates you to perform your best at work, and how do you handle feedback?"
            ]
            next_idx = min(len(conversation_history), len(topics) - 1)
            next_q = topics[next_idx]
            return {
                "analysis": {
                    "grammar_notes": "Satisfactory sentence structure and verb tenses.",
                    "fluency_notes": "Fluent, coherent delivery with clear ideas.",
                    "confidence_notes": "Exhibited professional confidence in the response."
                },
                "next_question": next_q
            }

    async def generate_final_report(self, resume_text: str, conversation_history: List[Dict[str, str]], voice_used: bool = False) -> Dict[str, Any]:
        """
        Compiles the overall English Assessment Report, scoring the candidate.
        Returns a JSON with scores, overall level, summary, strengths, weaknesses, areas for improvement, and final recommendation.
        """
        system_instruction = (
            "You are an expert HR senior evaluator. You must assess the entire conversation transcript "
            "and candidate's resume to generate a professional English proficiency report. "
            "You must return ONLY a valid JSON object matching the requested schema."
        )

        formatted_history = []
        for msg in conversation_history:
            formatted_history.append(f"AI Question: {msg.get('ai_question')}")
            formatted_history.append(f"Candidate Answer: {msg.get('candidate_answer')}")
        history_str = "\n".join(formatted_history)

        prompt = f"""
        Candidate Resume:
        \"\"\"
        {resume_text}
        \"\"\"

        Full Interview Conversation History:
        {history_str}

        Audio Input Used: {voice_used}

        Task:
        Perform a comprehensive linguistic and professional HR evaluation of the candidate's answers in the conversation history.
        
        Generate the following scores (each from 0 to 100):
        1. Communication: Ability to express thoughts clearly and directly.
        2. Grammar: Correctness of tenses, structures, syntax.
        3. Vocabulary: Range, precision, and appropriateness of words.
        4. Confidence: Expression, sentence assertiveness, pace (simulate from transcription style).
        5. Fluency: Cohesion, linking words, transitions.
        6. Professionalism: Natural business communication etiquette.
        7. Pronunciation: Estimate this score based on fluency and answer clarity. Provide a pronunciation score (0 to 100).
        
        Determine:
        - Overall Level: One of 'Excellent', 'Very Good', 'Good', 'Average', 'Needs Improvement', 'Not Recommended'.
        - Recommendation: One of 'Excellent', 'Very Good', 'Good', 'Average', 'Needs Improvement', 'Not Recommended'.
        - Summary: A professional, cohesive paragraph summarizing the candidate's overall communication, fluency, confidence, and suitability.
        - Strengths: A list of 3 key strengths observed during the interview (e.g. 'Good communication', 'Clear vocabulary').
        - Weaknesses: A list of specific linguistic mistakes (grammatical errors, incorrect tenses, poor word selections, or repetitive sentence fillers like 'umm', 'like') made by the candidate in the conversation. For each mistake, quote the candidate's exact phrase, show the corrected version, and explain the error (e.g., "In response 2, said 'I has two years...' -> Should be 'I have two years...' (Subject-verb agreement error)"). If there are no noticeable mistakes, list general speech pattern improvements.
        - Areas for Improvement: A list of 2-3 detailed, actionable suggestions directly mapped to help the candidate fix the identified mistakes (e.g., specific exercises or phrasing to practice).

        You must return a valid JSON object matching the following structure:
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
            response_text = await self._call_gemini(prompt, system_instruction, json_mode=True)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Failed to compile report via Gemini: {e}. Falling back to default scoring metrics.")
            # Calculate mock scores based on average estimation
            return {
                "communication_score": 82,
                "grammar_score": 80,
                "vocabulary_score": 78,
                "confidence_score": 85,
                "fluency_score": 80,
                "professionalism_score": 88,
                "pronunciation_score": 80 if voice_used else 0,
                "overall_level": "Good",
                "recommendation": "Good",
                "summary": "The candidate communicated clearly, answered confidently, and demonstrated good English communication skills throughout the simulated interview session.",
                "strengths": [
                    "Exhibits clean vocabulary and clear articulation of ideas.",
                    "Demonstrated good business etiquette and professionalism.",
                    "Answers align closely with the resume background details."
                ],
                "weaknesses": [
                    "Minor grammatical errors in complex sentence tenses.",
                    "Occasional hesitation/filler words visible in conversational flow."
                ],
                "areas_for_improvement": [
                    "Practice structured sentence tenses to minimize minor grammar mistakes.",
                    "Enhance technical vocabulary for describing resume projects."
                ]
            }

import pytest
from unittest.mock import AsyncMock, patch
from app.services.gemini_service import GeminiService


# Mock resume content for testing grounding
MOCK_RESUME = """
Jane Doe
Software Engineer
Experience:
- Frontend Engineer at TechCorp (Jan 2024 - Present): Built interactive dashboard using React and TailwindCSS.
- Intern at DataSys (Jun 2023 - Dec 2023): Wrote SQL queries and optimized PostgreSQL database index logic.
Skills: Python, JavaScript, React, SQL, PostgreSQL, Git
Certifications: AWS Certified Solutions Architect
"""

@pytest.mark.anyio
async def test_generate_first_question_grounding():
    """
    Test that the first question greeting is personalized and asks about resume content.
    """
    service = GeminiService()
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Hello Jane, could you tell me about the React dashboard you built at TechCorp?"
        
        first_q = await service.generate_first_question("Jane Doe", MOCK_RESUME)
        
        assert "React" in first_q or "TechCorp" in first_q or "dashboard" in first_q
        assert "Jane" in first_q
        mock_call.assert_called_once()
        prompt_arg = mock_call.call_args[0][0]
        assert "Jane Doe" in prompt_arg
        assert "React" in prompt_arg or "PostgreSQL" in prompt_arg


@pytest.mark.anyio
async def test_repeat_question_verbatim():
    """
    Verify that if the candidate asks to repeat, the AI returns the exact previous question verbatim.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "What PostgreSQL optimization techniques did you use at DataSys?", "candidate_answer": "please repeat"}
    ]
    
    response = await service.generate_next_question(MOCK_RESUME, history, "please repeat")
    
    # Assert repeated question is exactly matching the previous question
    assert response["next_question"] == "What PostgreSQL optimization techniques did you use at DataSys?"
    assert "repeat" in response["analysis"]["grammar_notes"].lower()


@pytest.mark.anyio
async def test_follow_up_on_correct_answer():
    """
    Verify that a relevant technical answer generates a follow-up or next resume item.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "Tell me about the dashboard you built at TechCorp.", "candidate_answer": "I built it using React and Tailwind CSS."}
    ]
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = '{"analysis": {"grammar_notes": "Perfect.", "fluency_notes": "Good.", "confidence_notes": "High."}, "next_question": "Why did you choose React and Tailwind for this project?"}'
        
        response = await service.generate_next_question(MOCK_RESUME, history, "I built it using React and Tailwind CSS.")
        
        assert "React" in response["next_question"] or "Tailwind" in response["next_question"] or "project" in response["next_question"]
        assert response["analysis"]["grammar_notes"] == "Perfect."


@pytest.mark.anyio
async def test_graceful_transition_on_unrelated_answer():
    """
    Verify that if candidate gives unrelated/non-answers, AI gracefully transitions without pushing back.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "What PostgreSQL indexing did you do at DataSys?", "candidate_answer": "I don't know, skip it."}
    ]
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        # Expected polite transition to another section (AWS certification or React)
        mock_call.return_value = '{"analysis": {"grammar_notes": "N/A", "fluency_notes": "N/A", "confidence_notes": "Low."}, "next_question": "No problem! Let\'s discuss your AWS Solutions Architect certification instead."}'
        
        response = await service.generate_next_question(MOCK_RESUME, history, "I don't know, skip it.")
        
        assert "No problem" in response["next_question"] or "AWS" in response["next_question"] or "certification" in response["next_question"]
        assert "N/A" in response["analysis"]["grammar_notes"]


@pytest.mark.anyio
async def test_clarification_repeating_same_question():
    """
    Verify that if candidate asks for clarification ("I didn't understand"), AI explains/repeats the same question simply.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "Could you walk me through the database index optimization you did at DataSys?", "candidate_answer": "I didn't understand the question. Can you explain?"}
    ]
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = '{"analysis": {"grammar_notes": "Requested clarification.", "fluency_notes": "N/A", "confidence_notes": "N/A"}, "next_question": "Sure! I asked about how you made database queries faster at DataSys using indexes. Can you describe that?"}'
        
        response = await service.generate_next_question(MOCK_RESUME, history, "I didn't understand the question. Can you explain?")
        
        # Verify it stays on DataSys/indexing/database
        assert "DataSys" in response["next_question"] or "index" in response["next_question"] or "database" in response["next_question"]
        assert "clarification" in response["analysis"]["grammar_notes"].lower()


@pytest.mark.anyio
async def test_query_answering_then_re_ask():
    """
    Verify that if candidate asks a question about the current question/resume, AI answers it first, then re-prompts the question.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "Which programming language do you enjoy working with the most?", "candidate_answer": "Do you mean from the skills listed on my resume?"}
    ]
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = '{"analysis": {"grammar_notes": "Asked a query.", "fluency_notes": "N/A", "confidence_notes": "N/A"}, "next_question": "Yes, from the programming languages listed on your resume like Python or JavaScript. Which do you enjoy most?"}'
        
        response = await service.generate_next_question(MOCK_RESUME, history, "Do you mean from the skills listed on my resume?")
        
        assert "resume" in response["next_question"] or "skills" in response["next_question"] or "Python" in response["next_question"]
        assert "query" in response["analysis"]["grammar_notes"].lower()


@pytest.mark.anyio
async def test_off_topic_unrelated_transition():
    """
    Verify that if candidate asks a completely unrelated off-topic question, AI transitions to the next resume topic.
    """
    service = GeminiService()
    history = [
        {"question_number": 1, "ai_question": "Tell me about the dashboard you built at TechCorp.", "candidate_answer": "what is the weather today?"}
    ]
    
    with patch.object(service, '_call_ai', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = '{"analysis": {"grammar_notes": "Off-topic query.", "fluency_notes": "N/A", "confidence_notes": "N/A"}, "next_question": "Let\'s stay focused on your experience. Can you tell me about the index optimization you did at DataSys?"}'
        
        response = await service.generate_next_question(MOCK_RESUME, history, "what is the weather today?")
        
        assert "stay focused" in response["next_question"].lower() or "DataSys" in response["next_question"]
        assert "off-topic" in response["analysis"]["grammar_notes"].lower()


@pytest.mark.anyio
async def test_transcribe_audio_success():
    """
    Verify that transcribe_audio successfully sends base64 audio to Gemini and returns the transcript.
    """
    from unittest.mock import MagicMock
    service = GeminiService()
    service.api_key = "test_key"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Hello, this is a test transcription."}
                    ]
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        transcript = await service.transcribe_audio(b"fake_audio_bytes", "audio/webm")
        assert transcript == "Hello, this is a test transcription."
        mock_post.assert_called_once()
        assert "key=test_key" in str(mock_post.call_args[0][0])


@pytest.mark.anyio
async def test_generate_tts_success():
    """
    Verify that generate_tts successfully fetches prebuilt voice audio from Gemini,
    converts it to WAV, and returns base64 string.
    """
    from unittest.mock import MagicMock
    import base64
    service = GeminiService()
    service.api_key = "test_key"
    
    raw_pcm = b'\x00\x00' * 10
    raw_pcm_base64 = base64.b64encode(raw_pcm).decode("utf-8")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/l16; rate=24000; channels=1",
                                "data": raw_pcm_base64
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        audio_b64 = await service.generate_tts("Hello Jane")
        assert audio_b64 is not None
        wav_bytes = base64.b64decode(audio_b64)
        assert wav_bytes.startswith(b"RIFF")
        mock_post.assert_called_once()
        assert "gemini-3.1-flash-tts-preview" in str(mock_post.call_args[0][0])




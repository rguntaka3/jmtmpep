import json
import os
from groq import Groq

async def evaluate_generated_question(question_stem: str, options: list, correct_answer: str, explanation: str, source_context: str, chapter_context: str = "") -> dict:
    """
    Uses an independent cross-model LLM judge (Qwen) to evaluate the quality, 
    faithfulness, legal accuracy, and section heading validity of a generated MPEP practice question,
    strictly flagging placeholder hallucinations (e.g., '100.XX').
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {
            "is_valid": False,
            "faithfulness_score": 1.0,
            "distractor_score": 1.0,
            "explanation_score": 1.0,
            "feedback": "Evaluation error: GROQ_API_KEY environment variable is not set."
        }

    client = Groq(api_key=api_key)

    judge_system_prompt = (
        "You are an expert USPTO Patent Bar exam proctor and legal education quality assurance judge. "
        "Your task is to evaluate a generated MPEP practice question for accuracy, distractor quality, explanation clarity, "
        "and crucially, **MPEP Section Heading/Citation Validity**.\n\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "1. Check the chapter context, title, and hint for placeholder or fabricated section numbers (e.g., '100.XX', '700.XX', or generic unverified section codes). \n"
        "2. If the section heading contains placeholders like '.XX' or references an MPEP section not supported by authentic USPTO guidelines, you MUST set 'is_valid' to false.\n"
        "3. Ensure the explanation explicitly cites real, verifiable MPEP chapters and sections.\n\n"
        "You must return your evaluation strictly in valid JSON format with the following keys:\n"
        "- 'is_valid': boolean (true only if overall score >= 4.0, zero hallucinations, and accurate section citations)\n"
        "- 'faithfulness_score': float (1.0 to 5.0, checking alignment with the MPEP source context)\n"
        "- 'distractor_score': float (1.0 to 5.0, checking if incorrect options are plausible legal traps)\n"
        "- 'explanation_score': float (1.0 to 5.0, checking clarity and citation correctness)\n"
        "- 'feedback': string (constructive critique, specifically calling out any section number hallucinations if found)"
    )

    user_prompt = (
        f"Chapter / Section Context Header: {chapter_context}\n"
        f"Source MPEP Context:\n{source_context}\n\n"
        f"Generated Question Stem:\n{question_stem}\n\n"
        f"Options:\n" + "\n".join(options) + "\n\n"
        f"Correct Answer: {correct_answer}\n"
        f"Explanation:\n{explanation}\n\n"
        "Evaluate this item strictly according to the system instructions, verifying that there are no section heading hallucinations (like .XX), and output valid JSON only."
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800  # Keeps output tokens safely below rate limits
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "is_valid": False,
            "faithfulness_score": 1.0,
            "distractor_score": 1.0,
            "explanation_score": 1.0,
            "feedback": f"Evaluation pipeline error: {str(e)}"
        }
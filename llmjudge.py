import json
import os
from groq import Groq

async def evaluate_generated_question(question_stem: str, options: list, correct_answer: str, explanation: str, source_context: str) -> dict:
    """
    Uses an independent cross-model LLM judge (Qwen) to evaluate the quality, 
    faithfulness, and legal accuracy of a generated MPEP practice question.
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
        "Your task is to evaluate a generated MPEP practice question for accuracy, distractor quality, and instructional value. "
        "You must return your evaluation strictly in valid JSON format with the following keys:\n"
        "- 'is_valid': boolean (true if overall score >= 4.0 and no factual errors)\n"
        "- 'faithfulness_score': float (1.0 to 5.0, checking alignment with the MPEP context)\n"
        "- 'distractor_score': float (1.0 to 5.0, checking if incorrect options are plausible legal traps)\n"
        "- 'explanation_score': float (1.0 to 5.0, checking clarity and citation correctness)\n"
        "- 'feedback': string (constructive critique or reason for failure)"
    )

    user_prompt = (
        f"Source MPEP Context:\n{source_context}\n\n"
        f"Generated Question Stem:\n{question_stem}\n\n"
        f"Options:\n" + "\n".join(options) + "\n\n"
        f"Correct Answer: {correct_answer}\n"
        f"Explanation:\n{explanation}\n\n"
        "Evaluate this item according to the system instructions and output valid JSON only."
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
            max_tokens=800  # Keeps output tokens safely below the 1000 OTPM rate limit
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
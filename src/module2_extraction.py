import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_claims(backstory_text):
    """
    Atomic Fact Extraction using Gemini 2.0 Flash (optimized for speed).
    
    Algorithm:
    - Breaks backstory into atomic facts (single propositions)
    - Validates factual nature (no opinions or speculation)
    - Structures output as JSON for downstream processing
    - Falls back to TF-based sentence scoring on API failure
    
    Returns: List of 5-15 concise, verifiable claims
    """
    prompt = f"""You are a fact extractor for literary consistency checking.

Extract ATOMIC FACTS from this backstory:
"{backstory_text}"

Requirements:
- Each fact = ONE claim (single proposition)
- Facts must be VERIFIABLE (checkable against text)
- Exclude opinions, feelings, speculation
- Focus on WHO, WHAT, WHERE, WHEN, HOW (factual elements)

Return valid JSON with this exact structure:
{{
  "facts": [
    "fact 1",
    "fact 2",
    ...
  ]
}}

Extract 5-15 facts. Be strict: only include verifiable claims."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        claims = result.get("facts", [])
        
        # Validate response
        if not isinstance(claims, list) or len(claims) == 0:
            print(f"Invalid API response, using fallback")
            return extract_claims_fallback(backstory_text)
        
        # Filter empty/whitespace claims
        claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
        return claims[:15]  # Cap at 15 claims
    
    except Exception as e:
        print(f"API Error in extract_claims: {e}")
        return extract_claims_fallback(backstory_text)

def extract_claims_fallback(text):
    """
    Fallback: sentence-level extraction scored by simple term diversity.
    Splits text into sentences, filters short ones, ranks by unique non-stopword ratio.
    """
    import re

    # Split into sentences
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]

    # Basic stopword set for English
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "and", "in", "on", "for", "with", "as", "by"
    }

    scored = []
    for sent in sentences:
        tokens = [w.strip(".,;:!?") for w in sent.lower().split()]
        words = [w for w in tokens if w not in stopwords and len(w) > 2]
        unique_ratio = len(set(words)) / (len(tokens) + 1)
        scored.append((unique_ratio, sent))

    # Return top 15 sentences as claims
    top = [s for _, s in sorted(scored, reverse=True)[:15]]
    # Ensure non-empty list
    return top if top else sentences[:5]
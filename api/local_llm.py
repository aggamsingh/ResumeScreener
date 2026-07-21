import logging
import os
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_model = None
_tokenizer = None

def get_local_llm():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        logger.info("Initializing local LLM: %s (first run will download ~950MB to data/models_cache)...", _LOCAL_LLM_MODEL)
        try:
            cache_dir = Path("./data/models_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            _tokenizer = AutoTokenizer.from_pretrained(
                _LOCAL_LLM_MODEL, 
                cache_dir=str(cache_dir)
            )
            _model = AutoModelForCausalLM.from_pretrained(
                _LOCAL_LLM_MODEL,
                cache_dir=str(cache_dir),
                low_cpu_mem_usage=True
            )
            logger.info("Local LLM %s loaded successfully!", _LOCAL_LLM_MODEL)
        except Exception as e:
            logger.error("Failed to load local LLM: %s", e, exc_info=True)
            raise e
    return _model, _tokenizer

def generate_local_jd(job_title: str) -> dict:
    try:
        model, tokenizer = get_local_llm()
        
        prompt = (
            f"Write a professional Job Description & Requirements for the role: '{job_title}'. "
            f"Provide a role description (2 paragraphs), followed by 4 bullet points of key responsibilities. "
            f"Also, identify 6-8 core skill keywords for this job.\n\n"
            f"Format the output strictly as a JSON object with keys 'job_description' and 'keywords' (array of strings)."
        )
        
        messages = [
            {"role": "system", "content": "You are a professional recruitment assistant. You must output only valid JSON matching the requested schema."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=350,
            temperature=0.7,
            do_sample=True
        )
        
        # Strip input tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Try to parse JSON
        try:
            cleaned = response.strip()
            # Strip markdown fences if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Simple fallback regex if JSON parser complains about trailing comma
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            
            data = json.loads(cleaned)
            if "job_description" in data and "keywords" in data:
                return data
        except Exception as json_err:
            logger.warning("Failed to parse JSON from local LLM output: %s. Raw output: %s", json_err, response)
            
        # Fallback parser if JSON parse fails
        return {
            "job_description": response,
            "keywords": [job_title, "Communication", "Technical Skills"]
        }
    except Exception as e:
        logger.error("Error generating local JD: %s", e)
        return {
            "job_description": f"We are seeking a qualified {job_title} to join our team. The candidate will manage core responsibilities, collaborate with stakeholders, and ensure top performance.",
            "keywords": [job_title, "Professionalism", "Teamwork"]
        }

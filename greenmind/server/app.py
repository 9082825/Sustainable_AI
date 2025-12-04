import math
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- FastAPI app ----------
app = FastAPI(title="GreenMind API", version="1.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data models ----------
class Params(BaseModel):
    layers: int
    training_time_hours: float
    flops_per_hour: float

class PromptRequest(BaseModel):
    role: str
    context: str
    expectations: str
    params: Params

class SimplePrompt(BaseModel):
    role: str
    context: str
    expectations: str

# ---------- Core utilities ----------
def tokenize_count(s: str) -> int:
    return len(s.strip().split())

def prompt_features(role: str, context: str, expectations: str):
    tokens = tokenize_count(role) + tokenize_count(context) + tokenize_count(expectations)
    punctuation_complexity = len([ch for ch in context if ch in ";:,."])
    complexity = min(1.0, (0.25 + math.log10(max(tokens, 1)) * 0.15 + punctuation_complexity * 0.01))
    return {"tokens": tokens, "complexity": complexity, "sections": 3}

def estimate_kwh(features, params: Params):
    utilization = 0.6
    flops_to_kwh = 1e-12
    base_kwh = params.flops_per_hour * utilization * params.training_time_hours * flops_to_kwh
    seq_factor = 1 + math.log10(max(1, features["tokens"])) * 0.25
    complexity_factor = 1 + features["complexity"] * 0.3
    layer_factor = 1 + (params.layers - 1) * 0.15
    return base_kwh * seq_factor * complexity_factor * layer_factor

# ---------- Heuristic compressor ----------
STOPWORDS = set("""
a an the and or but so to of for in on with at by from as that which who whom whose
be am is are was were been being have has had do does did will would should can could
""".split())

def compact_sentence(s: str) -> str:
    s = " ".join(s.split()).strip()
    s = s.replace(" in order to ", " to ").replace(" step-by-step ", " stepwise ")
    for w in ["please ", " kindly ", " basically ", " actually ", " really "]:
        s = s.replace(w, " ")
    return " ".join(s.split())

def to_bullets(text: str) -> str:
    parts = [p.strip() for p in text.replace(";", ".").split(".") if p.strip()]
    bullets = []
    for p in parts:
        tokens = [t for t in p.split() if t.lower() not in STOPWORDS]
        if tokens:
            bullets.append("- " + " ".join(tokens))
    return "\n".join(bullets[:4])

def abbreviate_phrases(s: str) -> str:
    abbr = {
        "Provide": "Give", "Return": "Deliver",
        "include": "add", "avoid": "skip",
        "responses": "answers",
        "concise, accurate": "concise & accurate",
        "step-by-step reasoning": "stepwise reasoning"
    }
    for k, v in abbr.items():
        s = s.replace(k, v)
    return s

def compress_prompt(role: str, context: str, expectations: str, max_tokens: int | None = None):
    r = abbreviate_phrases(compact_sentence(role))
    c = compact_sentence(context)
    e = compact_sentence(expectations)
    c_bullets = to_bullets(c)
    e_bullets = to_bullets(e)
    candidate = {
        "role": r,
        "context": c_bullets or c,
        "expectations": e_bullets or e
    }

    def count_tokens(pr):
        return tokenize_count(pr["role"]) + tokenize_count(pr["context"]) + tokenize_count(pr["expectations"])

    def remove_shortest(lines):
        if len(lines) <= 1:
            return lines
        lengths = [tokenize_count(l) for l in lines]
        idx = lengths.index(min(lengths))
        return lines[:idx] + lines[idx+1:]

    if max_tokens is not None:
        guard = 0
        while count_tokens(candidate) > max_tokens and guard < 20:
            guard += 1
            ctx_lines = candidate["context"].split("\n") if "\n" in candidate["context"] else [candidate["context"]]
            exp_lines = candidate["expectations"].split("\n") if "\n" in candidate["expectations"] else [candidate["expectations"]]
            if len(exp_lines) > 1:
                exp_lines = remove_shortest(exp_lines)
                candidate["expectations"] = "\n".join([l for l in exp_lines if l.strip()])
            elif len(ctx_lines) > 1:
                ctx_lines = remove_shortest(ctx_lines)
                candidate["context"] = "\n".join([l for l in ctx_lines if l.strip()])
            else:
                candidate["role"] = abbreviate_phrases(candidate["role"])
                candidate["context"] = abbreviate_phrases(candidate["context"])
                candidate["expectations"] = abbreviate_phrases(candidate["expectations"])
                break

    return candidate

# ---------- NLP module (embeddings + T5) ----------
try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
except Exception as e:
    embed_model = None
    t5_tokenizer = None
    t5_model = None
    print("NLP models not loaded:", e)

def semantic_similarity(a: str, b: str) -> float:
    try:
        if embed_model is None:
            ta = set(a.lower().split()); tb = set(b.lower().split())
            inter = len(ta & tb); return inter / max(1, len(ta))
        emb1 = embed_model.encode(a, convert_to_tensor=True)
        emb2 = embed_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except Exception:
        return 0.0

def t5_simplify(text: str, max_len: int = 128) -> str:
    try:
        if t5_model is None or t5_tokenizer is None:
            return text
        input_ids = t5_tokenizer("paraphrase: " + text, return_tensors="pt").input_ids
        outputs = t5_model.generate(input_ids, max_length=max_len, num_beams=4, early_stopping=True)
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return text

# ---------- Endpoints ----------

# Check Prompt → baseline analysis
@app.post("/analyze")
def analyze(req: PromptRequest):
    original_text = f"{req.role}\n{req.context}\n{req.expectations}"
    features = prompt_features(req.role, req.context, req.expectations)
    kwh = float(estimate_kwh(features, req.params))
    return {
        "original": original_text,
        "energy": kwh,
        "features": features
    }

# Make It Better → energy-aware refinement
@app.post("/improve")
def improve(req: PromptRequest):
    original_text = f"{req.role}\n{req.context}\n{req.expectations}"
    before_features = prompt_features(req.role, req.context, req.expectations)
    before_kwh = float(estimate_kwh(before_features, req.params))

    reduction_pct = 0.4
    target_tokens = max(1, int(before_features["tokens"] * (1 - reduction_pct)))

    # Candidate 1: heuristic compression
    cand_heur = compress_prompt(req.role, req.context, req.expectations, target_tokens)

    # Candidate 2: T5 paraphrase + compression
    t5_out = t5_simplify(original_text, max_len=128)
    parts = t5_out.split("\n")
    role_t5 = parts[0] if parts else req.role
    context_t5 = parts[1] if len(parts) > 1 else req.context
    expectations_t5 = parts[2] if len(parts) > 2 else req.expectations
    cand_t5 = compress_prompt(role_t5, context_t5, expectations_t5, target_tokens)

    pool = [cand_heur, cand_t5]

    best = None
    best_score = (float("inf"), float("inf"), float("inf"))

    for cand in pool:
        cand_text = f"{cand['role']}\n{cand['context']}\n{cand['expectations']}"
        sim = float(semantic_similarity(original_text, cand_text))
        af = prompt_features(cand["role"], cand["context"], cand["expectations"])
        akwh = float(estimate_kwh(af, req.params))
        score = (akwh, -sim, af["tokens"])
        if sim >= 0.85 and score < best_score:
            best = {"improved": cand, "after_features": af, "after_kwh": akwh, "sim": sim}
            best_score = score

    # Fallback if none pass similarity
    if best is None:
        cand = cand_heur
        cand_text = f"{cand['role']}\n{cand['context']}\n{cand['expectations']}"
        sim = float(semantic_similarity(original_text, cand_text))
        af = prompt_features(cand["role"], cand["context"], cand["expectations"])
        akwh = float(estimate_kwh(af, req.params))
        best = {"improved": cand, "after_features": af, "after_kwh": akwh, "sim": sim}

    improved_text = f"{best['improved']['role']}\n{best['improved']['context']}\n{best['improved']['expectations']}"

    return {
        "original": original_text,
        "improved": improved_text,
        "similarity": float(best["sim"]),
        "predicted_kwh_before": before_kwh,
        "predicted_kwh_after": float(best["after_kwh"]),
        "features_before": before_features,
        "features_after": best["after_features"]
    }

# Make It Clearer → NLP simplification
@app.post("/nlp_optimize")
def nlp_optimize(req: SimplePrompt):
    original_text = f"{req.role}\n{req.context}\n{req.expectations}".strip()

    # T5 paraphrase fallback-safe
    simplified = t5_simplify(original_text, max_len=128) if original_text else ""
    if not simplified:
        simplified = original_text

    # Similarity fallback-safe
    sim = float(semantic_similarity(original_text, simplified)) if original_text else 0.0

    return {
        "original": original_text,
        "simplified": simplified,
        "similarity": sim
    }

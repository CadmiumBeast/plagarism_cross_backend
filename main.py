from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import zipfile
import os
import difflib
from collections import defaultdict
from itertools import combinations, groupby

import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_python as tspy
from tree_sitter import Language, Parser


# ---------------------------------------------------------------------------
# 1. App (single instance)
# ---------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Plagiarism detector API is running"}


# ---------------------------------------------------------------------------
# 2. AST + IR
# ---------------------------------------------------------------------------
JAVA_LANGUAGE   = Language(tsjava.language())
CSHARP_LANGUAGE = Language(tscsharp.language())
PYTHON_LANGUAGE = Language(tspy.language())

LANGUAGE_MAP = {
    ".java": JAVA_LANGUAGE,
    ".cs":   CSHARP_LANGUAGE,
    ".py":   PYTHON_LANGUAGE,
}

IR_MAP = {
    "class_declaration": "CLASS_DEF", "class_definition": "CLASS_DEF",
    "interface_declaration": "INTERFACE_DEF",
    "method_declaration": "FUNC_DEF", "constructor_declaration": "FUNC_DEF",
    "function_definition": "FUNC_DEF", "arrow_function": "FUNC_DEF",
    "local_function_statement": "FUNC_DEF", "lambda_expression": "LAMBDA",
    "lambda": "LAMBDA",
    "method_invocation": "FUNC_CALL", "invocation_expression": "FUNC_CALL",
    "call": "FUNC_CALL", "object_creation_expression": "FUNC_CALL",
    "return_statement": "RETURN", "return": "RETURN",
    "local_variable_declaration": "VAR_DECL", "variable_declaration": "VAR_DECL",
    "variable_declarator": "VAR_ASSIGN", "assignment_expression": "VAR_ASSIGN",
    "assignment": "VAR_ASSIGN", "augmented_assignment": "VAR_ASSIGN",
    "if_statement": "IF", "if": "IF", "else_clause": "ELSE", "else": "ELSE",
    "elif_clause": "ELSE", "switch_statement": "SWITCH",
    "switch_expression": "SWITCH", "match_statement": "SWITCH",
    "for_statement": "LOOP_FOR", "enhanced_for_statement": "LOOP_FOREACH",
    "for_each_statement": "LOOP_FOREACH", "for": "LOOP_FOREACH",
    "while_statement": "LOOP_WHILE", "while": "LOOP_WHILE",
    "do_statement": "LOOP_DO_WHILE",
    "break_statement": "BREAK", "break": "BREAK",
    "continue_statement": "CONTINUE", "continue": "CONTINUE",
    "try_statement": "TRY", "try": "TRY", "catch_clause": "CATCH",
    "except_clause": "CATCH", "finally_clause": "FINALLY", "finally": "FINALLY",
    "binary_expression": "BINARY_OP", "binary_operator": "BINARY_OP",
    "boolean_operator": "BINARY_OP", "comparison_operator": "BINARY_OP",
    "unary_expression": "UNARY_OP", "unary_operator": "UNARY_OP",
    "not_operator": "UNARY_OP",
    "array_access": "ARRAY_ACCESS", "element_access_expression": "ARRAY_ACCESS",
    "subscript": "ARRAY_ACCESS",
    "string_literal": "LIT_STRING", "string": "LIT_STRING",
    "integer_literal": "LIT_INT",   "integer": "LIT_INT",
    "real_literal": "LIT_FLOAT",    "float": "LIT_FLOAT",
    "boolean": "LIT_BOOL", "true": "LIT_BOOL", "false": "LIT_BOOL",
    "null_literal": "LIT_NULL", "none": "LIT_NULL",
    "import_statement": "IMPORT", "using_directive": "IMPORT",
    "import_declaration": "IMPORT",
}


def _traverse(node, tokens: list):
    t = IR_MAP.get(node.type)
    if t:
        tokens.append(t)
    for child in node.children:
        _traverse(child, tokens)


def get_ir(source_code: str, ext: str) -> str:
    lang_obj = LANGUAGE_MAP.get(ext)
    if not lang_obj or not source_code.strip():
        return ""
    try:
        parser = Parser(lang_obj)
        tree   = parser.parse(bytes(source_code, "utf-8"))
        tokens: list[str] = []
        _traverse(tree.root_node, tokens)
        return " ".join(tokens)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
class SiameseVerifier(nn.Module):
    def __init__(self, emb_dim: int = 768, dropout_1: float = 0.3, dropout_2: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4 + 1, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_1),
            nn.Linear(512, 128),             nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout_2),
            nn.Linear(128, 32),              nn.GELU(),
            nn.Linear(32, 1),               nn.Sigmoid(),
        )

    def forward(self, u, v, ir_sim):
        interaction = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        fused       = torch.cat([interaction, ir_sim], dim=-1)
        return self.mlp(fused)


# ---------------------------------------------------------------------------
# 4. Model + tokenizer init
# ---------------------------------------------------------------------------
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
unixcoder = AutoModel.from_pretrained("microsoft/unixcoder-base").to(DEVICE)
unixcoder.eval()

model      = SiameseVerifier().to(DEVICE)
checkpoint = torch.load("siamese_best.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ---------------------------------------------------------------------------
# 5. In-memory cache  (populated by /analyze)
# ---------------------------------------------------------------------------
_file_cache:  dict[str, str] = {}   # file path → source code
_comparisons: list[dict]     = []   # all pairwise comparison results


# ---------------------------------------------------------------------------
# 6. Core helpers
# ---------------------------------------------------------------------------
def embed_code(source_code: str, ir_stream: str = "") -> torch.Tensor:
    text    = f"<IR> {ir_stream} </IR> <CODE> {source_code} </CODE>" if ir_stream else source_code
    encoded = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        out = unixcoder(**encoded)
    return F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1).squeeze().cpu()


def compare_files(file_a: str, code_a: str, file_b: str, code_b: str) -> dict:
    ext_a, ext_b = os.path.splitext(file_a)[1], os.path.splitext(file_b)[1]
    ir_a, ir_b   = get_ir(code_a, ext_a), get_ir(code_b, ext_b)

    ir_sim = (
        difflib.SequenceMatcher(None, ir_a.split(), ir_b.split(), autojunk=False).ratio()
        if ir_a and ir_b else 0.0
    )

    ir_sim_tensor = torch.tensor([[ir_sim]], dtype=torch.float32).to(DEVICE)
    vec_a = embed_code(code_a, ir_a).to(DEVICE)
    vec_b = embed_code(code_b, ir_b).to(DEVICE)

    with torch.no_grad():
        prob = model(vec_a.unsqueeze(0), vec_b.unsqueeze(0), ir_sim_tensor).item()

    combined = round((0.6 * prob + 0.4 * ir_sim) * 100, 2)
    severity = "high" if combined >= 75 else "medium" if combined >= 50 else "low"

    matcher = difflib.SequenceMatcher(None, code_a.splitlines(), code_b.splitlines(), autojunk=False)
    matching_blocks = [
        {"a_start": m.a, "b_start": m.b, "size": m.size}
        for m in matcher.get_matching_blocks()
        if m.size > 2
    ]

    return {
        "file_a":                file_a,
        "file_b":                file_b,
        "model_confidence":      round(prob * 100, 2),
        "structural_similarity": round(ir_sim * 100, 2),
        "combined_score":        combined,
        "severity":              severity,
        "flagged":               combined >= 55,
        "matching_blocks":       matching_blocks,
    }


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _merge_overlapping(matches: list[dict]) -> list[dict]:
    """Merge adjacent/overlapping line ranges per (other_student, other_file) pair."""
    key_fn = lambda m: (m["other_student"], m["other_file"])
    sorted_matches = sorted(matches, key=key_fn)
    merged: list[dict] = []

    for _, group in groupby(sorted_matches, key=key_fn):
        items   = sorted(group, key=lambda m: m["a_start"])
        current = None
        for item in items:
            if current is None:
                current = dict(item)
            elif item["a_start"] <= current["a_end"] + 1:
                current["a_end"] = max(current["a_end"], item["a_end"])
                current["b_end"] = max(current["b_end"], item["b_end"])
                if item["combined_score"] > current["combined_score"]:
                    current["combined_score"]        = item["combined_score"]
                    current["model_confidence"]      = item["model_confidence"]
                    current["structural_similarity"] = item["structural_similarity"]
                    current["severity"]              = item["severity"]
            else:
                merged.append(current)
                current = dict(item)
        if current is not None:
            merged.append(current)

    return sorted(merged, key=lambda m: m["a_start"])


# ---------------------------------------------------------------------------
# 7. /analyze
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_zip(file: UploadFile = File(...)):
    global _file_cache, _comparisons
    _file_cache  = {}
    _comparisons = []

    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb+") as f:
        f.write(file.file.read())

    extracted: dict[str, str] = {}
    with zipfile.ZipFile(tmp_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith((".java", ".py", ".cs")):
                try:    extracted[name] = zf.read(name).decode("utf-8")
                except: extracted[name] = zf.read(name).decode("latin-1")
    os.remove(tmp_path)
    _file_cache = extracted

    # Group by top-level folder = student
    student_files: dict[str, dict[str, str]] = defaultdict(dict)
    for path, code in extracted.items():
        parts   = path.replace("\\", "/").split("/")
        student = parts[0] if len(parts) > 1 else "__root__"
        student_files[student][path] = code

    students = list(student_files.keys())

    # Cross-student comparison
    comparisons: list[dict] = []
    for student_a, student_b in combinations(students, 2):
        files_a = student_files[student_a]
        files_b = student_files[student_b]
        pairs: list[dict] = []

        for fa, code_a in files_a.items():
            for fb, code_b in files_b.items():
                pairs.append(compare_files(fa, code_a, fb, code_b))

        pairs.sort(key=lambda r: r["combined_score"], reverse=True)
        scores = [p["combined_score"] for p in pairs]

        comparisons.append({
            "student_a":     student_a,
            "student_b":     student_b,
            "files_a":       list(files_a.keys()),
            "files_b":       list(files_b.keys()),
            "avg_score":     avg(scores),
            "max_score":     max(scores) if scores else 0.0,
            "flagged_count": sum(1 for p in pairs if p["flagged"]),
            "pairs":         pairs,
        })

    comparisons.sort(key=lambda c: c["max_score"], reverse=True)
    _comparisons = comparisons

    # Per-student summary
    student_summary: dict[str, dict] = {}
    for s in students:
        relevant_scores = [
            p["combined_score"]
            for comp in comparisons
            if comp["student_a"] == s or comp["student_b"] == s
            for p in comp["pairs"]
        ]
        student_summary[s] = {
            "files":     list(student_files[s].keys()),
            "max_score": max(relevant_scores) if relevant_scores else 0.0,
            "avg_score": avg(relevant_scores),
            "flagged":   any(sc >= 55 for sc in relevant_scores),
        }

    return {
        "status":      "success",
        "students":    student_summary,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# 8. /file
# ---------------------------------------------------------------------------
@app.get("/file")
async def get_file(name: str):
    code = _file_cache.get(name)
    if code is None:
        raise HTTPException(404, "File not found in cache. Re-run /analyze first.")
    return {"name": name, "code": code}


# ---------------------------------------------------------------------------
# 9. /file-matches
# ---------------------------------------------------------------------------
@app.get("/file-matches")
async def get_file_matches(student: str, file: str):
    code = _file_cache.get(file)
    if code is None:
        raise HTTPException(404, "File not found in cache. Re-run /analyze first.")

    matches: list[dict] = []

    for comp in _comparisons:
        if comp["student_a"] == student and file in comp["files_a"]:
            role          = "a"
            other_student = comp["student_b"]
        elif comp["student_b"] == student and file in comp["files_b"]:
            role          = "b"
            other_student = comp["student_a"]
        else:
            continue

        for pair in comp["pairs"]:
            if role == "a" and pair["file_a"] != file:
                continue
            if role == "b" and pair["file_b"] != file:
                continue

            other_file = pair["file_b"] if role == "a" else pair["file_a"]

            for block in pair["matching_blocks"]:
                if block["size"] == 0:
                    continue

                a_start = block["a_start"] if role == "a" else block["b_start"]
                b_start = block["b_start"] if role == "a" else block["a_start"]
                a_end   = a_start + block["size"] - 1
                b_end   = b_start + block["size"] - 1

                matches.append({
                    "a_start":               a_start,
                    "a_end":                 a_end,
                    "b_start":               b_start,
                    "b_end":                 b_end,
                    "other_student":         other_student,
                    "other_file":            other_file,
                    "combined_score":        pair["combined_score"],
                    "model_confidence":      pair["model_confidence"],
                    "structural_similarity": pair["structural_similarity"],
                    "severity":              pair["severity"],
                })

    return {
        "file":    file,
        "student": student,
        "code":    code,
        "matches": _merge_overlapping(matches),
    }
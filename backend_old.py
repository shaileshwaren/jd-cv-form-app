import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="JD→CV Scorer API")

BOT_PROMPT = r"""
version: 2.2
name: oxy_jd_cv_rubric_master_v2_mini
description: JD→CV scorer (compact). Parses JD, normalizes terms, weights 90/10, boosts must-haves, scores CVs 0–5, applies compliance & floor rules, outputs tables + decision.

COMMAND INPUT FLEXIBILITY
All commands now accept plain string input.
If a command receives a raw text input instead of JSON, the assistant MUST wrap it into the correct internal JSON object.

Rules:
- /evaluate
  If only a string is provided → treat it as a CV.
  Use latest rubric automatically unless rubric_key is specified.
  Format internally as:
    { "cv_texts": ["<string>"], "rubric_key": "<latest>" }

- /rescore
  Accept plain string CV input.
  Replace existing CV list with the new text(s).
  Use latest rubric automatically.

- /compare
  Accept 2+ CV texts separated by either:
    ---  OR  new lines
  Internally standardize to:
    { "candidates": ["cv1", "cv2", ...] }

- /export
  Export the most recently evaluated CV's full evaluation report.
  If a filename is not provided, auto-generate using:
    "<candidate_name>_AI_rpt"

- /generate rubric
  Already accepts raw JD text.
  Always output the generated rubric in yaml format.

metadata:
  language: en
  pass_threshold: 70
  borderline_band: {low: 65, high: 74}
  buckets: [compliance, must_have, nice_to_have]
  knowledge_ref: knowledge_rubric.yaml

/normalize_terms:
  parameters:
    ontology_ref: it_master_ontology.yaml
    enable_embeddings: true
    min_similarity: 0.82
    role_title_normalization: true
    prefer_specific_child: true
    guards: {java_not_javascript: true}

/boost:
  scope: must_have
  parameters: {targets: [keys], boost_pct: 10}
  action: [redistribute, apply_boost, renorm_90]

/boost_auto:
  scope: must_have
  parameters: {boost_pct: 10}
  description: >
    Auto-detect critical must-have items and boost their weights.
  action: [infer_targets_from_jd_or_rubric, apply_boost, renorm_90]

/generate_rubric:
  description: >
    Build a preset rubric from a JD and freeze it for reuse.
  parameters:
    jd_text: string
    role_applied_override: string|null
    rubric_name: string|null

inputs:
  jd_text: ""
  cv_texts: []
  rubric: null
  rubric_key: ""
  weights_override: {must_have: {}, nice_to_have: {}}

policies:
  bucket_rules:
    - Education & Years => COMPLIANCE ONLY
  rounding: 2_decimals
  compliance_overrides:
    years_of_experience: {derive_from_jd: true}
    work_authorization: {ignored: true}
  bias_guardrails:
    - Strip protected attributes

weights:
  buckets: {must_haves_total: 90, nice_to_haves_total: 10}
  distribution: {mode: auto_equal, validate_sums: true}

scoring:
  scale: {"0":"Absent","1":"Mention","2":"Basic","3":"Solid recent","4":"Strong impact","5":"Expert"}
  overall_pass_threshold: 70
  floor_rule: {enabled: true, rule: "any must-have < 2 => FAIL"}
  formula:
    per_item: "Weight% * (Score/5)"
    total: "sum(must_have) + sum(nice_to_have)"
  explainability:
    citations_required: true
    citation_format: "CV[line|bullet|section]"
    justification_line_max: 1

ensemble:
  enabled: true
  when: borderline_only
  generators: 3
  temperatures: [0.2,0.5,0.8]
  judge_criteria: ["Citations","Scale adherence","Consistency"]

jd_baseline_output:
  titles:
    must: "Must-have (sum weight = 90.00%)"
    nice: "Nice-to-have (sum weight = 10.00%)"
  columns: ["Requirement","Weight (%)","Score (0–5)","Wt Score (%)"]

global_presentation_rules:
  header_template:
    ["Candidate: {candidate_name}",
     "Role Applied: {role_applied}",
     "Final Score: {final_score} / 100"]

evaluation_pipeline:
  - step: jd_analysis
    actions:
      - If preset rubric exists: load and skip JD parsing
      - Else: extract requirements, assign weights, normalize, auto-boost
  - step: jd_baseline_tables
    actions: [render_tables_with_placeholders]
  - step: cv_scoring
    actions:
      - Score each CV with citations and justification
  - step: aggregation
    actions: [sum_buckets_round_2dp]
  - step: gates_and_floor
    actions:
      - Apply compliance gates
      - Apply floor rule
      - Apply borderline review flag
  - step: reflection_bias_check
    actions: [redact_protected_attrs]
  - step: ensemble_optional
    condition: "decision==REVIEW"
    actions: [run_generators, judge_select_best]
  - step: reporting
    actions:
      - Header, tables, totals, decision
      - Include citations + justification

reporting_format:
  header:
    title_lines:
      ["Candidate: {candidate_name}",
       "Role Applied: {role_applied}",
       "Final Score: {final_score} / 100"]
  tables:
    must_have_table:
      title: "Must-have (sum weight = 90.00%)"
      columns: ["Requirement","Weight (%)","Score (0–5)","Wt Score (%)","Citations"]
    nice_to_have_table:
      title: "Nice-to-have (sum weight = 10.00%)"
      columns: ["Acumen/Skills","Weight (%)","Score (0–5)","Wt Score (%)","Citations"]

validation_rules:
  - must_have weights sum to 90
  - nice_to_have weights sum to 10
  - education & years only in Compliance
  - floor rule must be applied
  - header must precede the must-have table
  - any non-zero score requires a citation
""".strip()

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore").replace("\t", "  ")
    return ""

def trim_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[:90_000]
    tail = text[-50_000:]
    return head + "\n\n... [TRUNCATED] ...\n\n" + tail

RUBRIC_TEXT = ""
for cand in [KNOWLEDGE_DIR / "knowledge_rubric.yaml", KNOWLEDGE_DIR / "knowledge_rubric.yml"]:
    RUBRIC_TEXT = read_text_if_exists(cand)
    if RUBRIC_TEXT:
        break

ONTOLOGY_TEXT = ""
for cand in [
    KNOWLEDGE_DIR / "it_master_ontology.yaml",
    KNOWLEDGE_DIR / "it_master_ontology.yml",
    KNOWLEDGE_DIR / "it_master_ontology_roles_skills_v_0_9.yaml",
    ]:
    ONTOLOGY_TEXT = read_text_if_exists(cand)
    if ONTOLOGY_TEXT:
        break

RUBRIC_TEMPLATE_TEXT = read_text_if_exists(KNOWLEDGE_DIR / "rubric_template.yaml")

RUBRIC_TEXT = trim_text(RUBRIC_TEXT, 140_000)
ONTOLOGY_TEXT = trim_text(ONTOLOGY_TEXT, 140_000)
RUBRIC_TEMPLATE_TEXT = trim_text(RUBRIC_TEMPLATE_TEXT, 120_000)

class GenerateRubricRequest(BaseModel):
    role_applied: str
    jd_text: str
    rubric_name: str | None = None

class ScoreRequest(BaseModel):
    role_applied: str
    output_format: str
    jd_text: str
    cvs: list[dict]
    preset_rubric_yaml: str | None = None

def build_generate_rubric_prompt(role_applied: str, jd_text: str, rubric_name: str | None) -> str:
    name_line = f"Rubric name: {rubric_name}" if rubric_name else "Rubric name: (auto)"
    return f"""
    SYSTEM BOT PROMPT (follow strictly):
    {BOT_PROMPT}

    REFERENCE KNOWLEDGE:
    --- knowledge_rubric.yaml ---
    {RUBRIC_TEXT}

    --- it_master_ontology.yaml ---
    {ONTOLOGY_TEXT}

    RUBRIC_TEMPLATE_YAML (use this template/structure when generating):
    {RUBRIC_TEMPLATE_TEXT}

    TASK:
    Generate a preset rubric from the JD.
    Output MUST be YAML only (no backticks).
    The rubric should be frozen/reusable as a preset.
    It must follow the template structure above.
    Ensure must-have weights sum to 90 and nice-to-have sum to 10.

    INPUTS:
    Role Applied Override: {role_applied}
    {name_line}

    JOB DESCRIPTION:
    {jd_text}
    """.strip()

def build_score_prompt(role_applied: str, output_format: str, jd_text: str, cvs: list[dict], preset_rubric_yaml: str | None) -> str:
    cv_blob = ""
    for i, cv in enumerate(cvs, start=1):
        cv_blob += f"\n\n=== Candidate {i}: {cv.get('name','CV')} ===\n{cv.get('text','')}\n"

    response_contract = """
    Return a SINGLE JSON object with these keys:
    - "markdown_report": string
    - "json_summary": object|null
    - If output_format == "JSON": include a structured summary object
    - Else: set to null
    - "report_data": object
    This MUST be a structured representation used to generate a formatted PDF.
    Use this schema exactly:

    {
        "title": "Candidate Scoring Report – <Role>",
        "candidate_name": "<string>",
        "role_applied": "<string>",
        "final_score": "<number as string or number>",
        "decision": "PASS|FAIL|REVIEW",
        "compliance": [
        {"item":"Education","status":"PASS|FAIL|NOT ASSESSED","details":"<string>"},
        {"item":"Years of Experience","status":"PASS|FAIL|NOT ASSESSED","details":"<string>"},
        {"item":"Work Authorization","status":"PASS|FAIL|NOT ASSESSED","details":"<string>"}
        ],
        "must_have": {
        "total_weight": 90.0,
        "subtotal_weighted": "<string like '78.47 / 90.00'>",
        "requirements": [
            {
            "idx": 1,
            "label": "<string>",
            "weight_pct": <number>,
            "score_0_5": <integer 0-5>,
            "weighted_contribution_pct": <number>,
            "evidence": "<string>"
            }
        ]
        },
        "nice_to_have": {
        "total_weight": 10.0,
        "subtotal_weighted": "<string like '10.00 / 10.00'>",
        "requirements": [
            {
            "idx": 1,
            "label": "<string>",
            "weight_pct": <number>,
            "score_0_5": <integer 0-5>,
            "weighted_contribution_pct": <number>,
            "evidence": "<string>"
            }
        ]
        },
        "totals": {
        "weighted_must_have": "<string like '78.47 / 90.00'>",
        "weighted_nice_to_have": "<string like '10.00 / 10.00'>",
        "total_weighted": "<string like '88.47 / 100.00'>",
        "floor_rule": "<string>",
        "compliance_summary": "<string>",
        "final_decision_line": "<string>",
        "rationale": "<string>"
        },
        "narrative": {
        "strengths": ["<string>", "<string>"],
        "gaps": ["<string>", "<string>"],
        "recommendation": "<string>"
        }
    }

    Do not wrap in triple backticks. Return valid JSON only.
    """.strip()


    preset_block = ""
    if preset_rubric_yaml and preset_rubric_yaml.strip():
        preset_block = f"""
        PRESET_RUBRIC_YAML (use this and skip JD parsing):
        {preset_rubric_yaml.strip()}
        """.strip()

    return f"""
    SYSTEM BOT PROMPT (follow strictly):
    {BOT_PROMPT}

    REFERENCE KNOWLEDGE:
    --- knowledge_rubric.yaml ---
    {RUBRIC_TEXT}

    --- it_master_ontology.yaml ---
    {ONTOLOGY_TEXT}

    {preset_block}

    USER INPUTS:
    Role Applied: {role_applied}

    JOB DESCRIPTION:
    {jd_text}

    CVS:
    {cv_blob}

    OUTPUT_FORMAT_FOR_UI_DISPLAY: {output_format}

    {response_contract}
    """.strip()

@app.post("/generate_rubric")
def generate_rubric(req: GenerateRubricRequest):
    if not req.role_applied.strip():
        return {"error": "Role applied is required."}
    if not req.jd_text or len(req.jd_text.strip()) < 20:
        return {"error": "JD text is missing or too short."}
    if not RUBRIC_TEMPLATE_TEXT.strip():
        return {"error": "rubric_template.yaml not found in /knowledge folder."}

    prompt = build_generate_rubric_prompt(req.role_applied.strip(), req.jd_text, req.rubric_name)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
    )

    return {"rubric_yaml": resp.output_text}

@app.post("/score")
def score(req: ScoreRequest):
    if not req.role_applied.strip():
        return {"error": "Role applied is required."}
    if not req.jd_text or len(req.jd_text.strip()) < 20:
        return {"error": "JD text is missing or too short."}
    if not req.cvs:
        return {"error": "No CVs provided."}

    prompt = build_score_prompt(
        role_applied=req.role_applied.strip(),
        output_format=req.output_format,
        jd_text=req.jd_text,
        cvs=req.cvs,
        preset_rubric_yaml=req.preset_rubric_yaml,
    )

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
    )

    return {"result": resp.output_text}

# version 1.2


import os
import json
from pathlib import Path
import streamlit as st
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
from docx import Document

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Circle, Rect, String
from reportlab.lib.enums import TA_LEFT

# Load local .env (only used on your computer)
load_dotenv()

# Read secrets from Streamlit Cloud first, fallback to local .env
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

api_key = get_secret("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
model = get_secret("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini"))


# Create OpenAI client
client = OpenAI(api_key=api_key)

# ----------------------------
# Bot prompt + Knowledge (merged from backend.py)
# ----------------------------
BOT_PROMPT = r"""
    version: 2.2
    name: oxy_jd_cv_rubric_master_v2_mini
    description: JD→CV scorer. Use rubric/ontology knowledge, score CVs 0–5, apply compliance + floor rule, output tables + decision.

    COMMAND INPUT FLEXIBILITY
    - /evaluate: if given a plain string, treat as a single CV: {"cv_texts":["<string>"], "rubric_key":"<latest>"}.
    - /rescore: accept plain string CV input; replace existing CV list; use latest rubric.
    - /compare: accept 2+ CV texts separated by '---' or newlines: {"candidates":["cv1","cv2",...]}.
    - /export: export most recently evaluated CV report; default filename "<candidate_name>_AI_rpt".
    - /generate rubric: output YAML only.

    POLICIES (mandatory)
    - Bias guardrails: strip protected attributes.
    - Compliance only (not scored): Education, Years of Experience.
    - Work authorization: ignored.
    - Rounding: 2 decimals.

    WEIGHTS
    - Buckets: must-have total = 90.00%, nice-to-have total = 10.00% (must sum exactly).

    SCORING
    - Scale 0–5:
    0 Absent; 1 Mention; 2 Basic; 3 Solid recent; 4 Strong impact; 5 Expert
    - Floor rule: any must-have < 2 ⇒ FAIL.
    - Any non-zero score requires a citation in the specified citation_format.
    - Explainability: keep justification to one line max.

    NORMALIZATION
    - Use ontology normalization (prefer specific child terms).
    - Guard: Java ≠ JavaScript.
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


# Load knowledge files once (per Streamlit session cache)
@st.cache_data(show_spinner=False)
def load_knowledge_texts() -> tuple[str, str, str]:
    rubric_text = ""
    for cand in [KNOWLEDGE_DIR / "knowledge_rubric.yaml", KNOWLEDGE_DIR / "knowledge_rubric.yml"]:
        rubric_text = read_text_if_exists(cand)
        if rubric_text:
            break

    ontology_text = ""
    for cand in [
        KNOWLEDGE_DIR / "it_master_ontology.yaml",
        KNOWLEDGE_DIR / "it_master_ontology.yml",
        KNOWLEDGE_DIR / "it_master_ontology_roles_skills_v_0_9.yaml",
    ]:
        ontology_text = read_text_if_exists(cand)
        if ontology_text:
            break

    rubric_template_text = read_text_if_exists(KNOWLEDGE_DIR / "rubric_template.yaml")

    rubric_text = trim_text(rubric_text, 140_000)
    ontology_text = trim_text(ontology_text, 140_000)
    rubric_template_text = trim_text(rubric_template_text, 120_000)

    return rubric_text, ontology_text, rubric_template_text


# Back-compat globals for prompt builders (keeps changes minimal)
RUBRIC_TEXT, ONTOLOGY_TEXT, RUBRIC_TEMPLATE_TEXT = load_knowledge_texts()


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


def build_score_prompt(role_applied: str, jd_text: str, cvs: list[dict], preset_rubric_yaml: str | None) -> str:
    preset_block = ""
    if preset_rubric_yaml and preset_rubric_yaml.strip():
        preset_block = f"""
            PRESET_RUBRIC_YAML (use this and skip JD parsing):
            {preset_rubric_yaml.strip()}
            """.strip()

    response_contract = """
        Return a SINGLE JSON object with exactly one top-level key:
        - "report_data": object

        The report_data MUST follow this schema exactly (no extra keys, no missing keys):

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

    cv_blob = ""
    for i, cv in enumerate(cvs, start=1):
        cv_blob += f"\n\n=== Candidate {i}: {cv.get('name','CV')} ===\n{cv.get('text','')}\n"

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

        {response_contract}
        """.strip()


# ----------------------------
# Helpers
# ----------------------------
def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.read()))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    if name.endswith(".docx"):
        doc = Document(BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs])

    return ""


def split_pasted_cvs(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("\n---\n") if p.strip()]
    return parts if parts else [raw]


def pdf_bytes_from_report_data(report_data: dict, markdown_fallback: str = "") -> bytes:
    """
    Generates the 'Version A Full Detail' styled PDF with:
    - colored traffic light dots
    - dots next to requirement titles
    - wrap-safe compliance table (no overflow)
    Falls back to plain text PDF if report_data is missing/invalid.
    """
    from io import BytesIO
    buffer = BytesIO()

    if not isinstance(report_data, dict) or not report_data:
        # fallback: simple monospace dump of markdown
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        x, y = 40, height - 40
        c.setFont("Courier", 9)
        for line in (markdown_fallback or "").splitlines():
            c.drawString(x, y, line[:110])
            y -= 12
            if y < 60:
                c.showPage()
                c.setFont("Courier", 9)
                y = height - 40
        c.save()
        return buffer.getvalue()

    # -------- Styles --------
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Heading1"], fontSize=20, leading=24, spaceAfter=10))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=10.2, leading=14))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9.2, leading=12.5))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10.2, leading=14, backColor=colors.whitesmoke, borderPadding=6))

    styles.add(ParagraphStyle(
        name="CellWrap",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=12,
        alignment=TA_LEFT,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="CellWrapBold",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=12,
        alignment=TA_LEFT,
        wordWrap="CJK",
    ))
    styles["CellWrapBold"].fontName = "Helvetica-Bold"

    def banner(text, fill):
        d = Drawing(540, 34)
        d.add(Rect(0, 0, 540, 34, fillColor=fill, strokeColor=fill))
        d.add(String(14, 11, text, fontName="Helvetica-Bold", fontSize=12, fillColor=colors.white))
        return d

    def traffic_color(score_0_5: int):
        if score_0_5 >= 4:
            return colors.HexColor("#22c55e")  # green
        if score_0_5 >= 2:
            return colors.HexColor("#f59e0b")  # amber
        return colors.HexColor("#ef4444")      # red

    def dot(score_0_5: int, diameter=10):
        d = Drawing(diameter, diameter)
        r = diameter / 2
        d.add(Circle(r, r, r, fillColor=traffic_color(score_0_5), strokeColor=colors.white, strokeWidth=0.5))
        return d

    def requirement_title_row(score_0_5: int, title_text: str):
        t = Table([[dot(score_0_5, 10), Paragraph(f"<b>{title_text}</b>", styles["BodyX"])]], colWidths=[14, 520])
        t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        return t

    # -------- Pull fields --------
    title = report_data.get("title") or f"Candidate Scoring Report – {report_data.get('role_applied','')}"
    candidate_name = report_data.get("candidate_name", "")
    role_applied = report_data.get("role_applied", "")
    final_score = report_data.get("final_score", "")
    decision = report_data.get("decision", "REVIEW")

    decision_color = colors.darkgreen if decision == "PASS" else (colors.red if decision == "FAIL" else colors.HexColor("#b45309"))

    compliance = report_data.get("compliance", []) or []
    must = report_data.get("must_have", {}) or {}
    nice = report_data.get("nice_to_have", {}) or {}
    totals = report_data.get("totals", {}) or {}
    narrative = report_data.get("narrative", {}) or {}

    # -------- Build PDF --------
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    story.append(Paragraph(title, styles["TitleX"]))

    header_tbl = Table(
        [
            ["Candidate:", candidate_name, "Final Score:", str(final_score)],
            ["Role Applied:", role_applied, "Decision:", decision],
        ],
        colWidths=[80, 220, 80, 140]
    )
    header_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 10.5),
        ("FONTNAME", (0,0), (0,1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,1), "Helvetica-Bold"),
        ("FONTNAME", (3,1), (3,1), "Helvetica-Bold"),
        ("TEXTCOLOR", (3,1), (3,1), decision_color),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 8))

    # 1) Compliance (wrap-safe table)
    story.append(banner("1. Compliance (Gating – Not Scored)", colors.HexColor("#444444")))
    story.append(Spacer(1, 8))

    comp_rows = [[
        Paragraph('<font color="white"><b>Item</b></font>', styles["CellWrapBold"]),
        "",
        Paragraph('<font color="white"><b>Status</b></font>', styles["CellWrapBold"]),
        Paragraph('<font color="white"><b>Detail</b></font>', styles["CellWrapBold"]),
    ]]

    for row in compliance:
        item = str(row.get("item", ""))
        status = str(row.get("status", "NOT ASSESSED")).upper()
        details = str(row.get("details", ""))

        if status == "PASS":
            dscore = 5
            status_txt = "PASS"
        elif status == "FAIL":
            dscore = 0
            status_txt = "FAIL"
        else:
            dscore = 3
            status_txt = "NOT ASSESSED"

        comp_rows.append([
            Paragraph(item, styles["CellWrap"]),
            dot(dscore, 10),
            Paragraph(status_txt, styles["CellWrap"]),
            Paragraph(details, styles["CellWrap"]),
        ])

    comp_tbl = Table(comp_rows, colWidths=[115, 18, 95, 312], repeatRows=1)
    comp_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (1,1), (1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#f2f2f2")]),
        ("WORDWRAP", (0,0), (-1,-1), "CJK"),
    ]))
    story.append(comp_tbl)
    story.append(Spacer(1, 10))

    # 2) Must-have
    story.append(banner("2. Must-have Requirements (Total Weight = 90.00%)", colors.HexColor("#444444")))
    story.append(Spacer(1, 8))

    for req in must.get("requirements", []) or []:
        idx = req.get("idx")
        label = req.get("label", "")
        weight = req.get("weight_pct", 0)
        score_0_5 = int(req.get("score_0_5", 0))
        contrib = req.get("weighted_contribution_pct", 0)
        evidence = req.get("evidence", "")

        block = []
        block.append(requirement_title_row(score_0_5, f"Requirement {idx}:"))
        block.append(Spacer(1, 2))
        block.append(Paragraph(f"<b>Label:</b> {label}", styles["BodyX"]))
        block.append(Paragraph(
            f"<b>Weight:</b> {float(weight):.2f}% &nbsp;&nbsp; "
            f"<b>Score (0–5):</b> {score_0_5} &nbsp;&nbsp; "
            f"<b>Weighted Score Contribution:</b> {float(contrib):.2f}%",
            styles["BodyX"]
        ))
        block.append(Paragraph(f"<b>Evidence:</b> {evidence}", styles["BodyX"]))
        block.append(Spacer(1, 6))
        block.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey))
        block.append(Spacer(1, 6))
        story.append(KeepTogether(block))

    if must.get("subtotal_weighted"):
        story.append(Paragraph(f"<b>Must-have subtotal (weighted):</b> {must.get('subtotal_weighted')}", styles["BodyX"]))
        story.append(Spacer(1, 10))

    # 3) Nice-to-have
    story.append(banner("3. Nice-to-have Requirements (Total Weight = 10.00%)", colors.HexColor("#444444")))
    story.append(Spacer(1, 8))

    for req in nice.get("requirements", []) or []:
        idx = req.get("idx")
        label = req.get("label", "")
        weight = req.get("weight_pct", 0)
        score_0_5 = int(req.get("score_0_5", 0))
        contrib = req.get("weighted_contribution_pct", 0)
        evidence = req.get("evidence", "")

        story.append(requirement_title_row(score_0_5, f"Requirement {idx}:"))
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"<b>Label:</b> {label}", styles["BodyX"]))
        story.append(Paragraph(
            f"<b>Weight:</b> {float(weight):.2f}% &nbsp;&nbsp; "
            f"<b>Score (0–5):</b> {score_0_5} &nbsp;&nbsp; "
            f"<b>Weighted Score Contribution:</b> {float(contrib):.2f}%",
            styles["BodyX"]
        ))
        story.append(Paragraph(f"<b>Evidence:</b> {evidence}", styles["BodyX"]))
        story.append(Spacer(1, 8))

    if nice.get("subtotal_weighted"):
        story.append(Paragraph(f"<b>Nice-to-have subtotal (weighted):</b> {nice.get('subtotal_weighted')}", styles["BodyX"]))
        story.append(Spacer(1, 10))

    # 4) Totals and Final Decision
    story.append(banner("4. Totals and Final Decision", colors.HexColor("#444444")))
    story.append(Spacer(1, 8))

    tot_tbl = Table([
        ["Weighted Must-have Score:", totals.get("weighted_must_have", "")],
        ["Weighted Nice-to-have Score:", totals.get("weighted_nice_to_have", "")],
        ["Total Weighted Score:", totals.get("total_weighted", "")],
        ["Floor Rule:", totals.get("floor_rule", "")],
        ["Compliance:", totals.get("compliance_summary", "")],
        ["Final Decision:", totals.get("final_decision_line", decision)],
    ], colWidths=[180, 360])
    tot_tbl.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.whitesmoke, colors.HexColor("#f2f2f2")]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    story.append(tot_tbl)
    story.append(Spacer(1, 10))

    rationale = totals.get("rationale", "")
    if rationale:
        story.append(Paragraph(f"<b>Rationale:</b> {rationale}", styles["Callout"]))
        story.append(Spacer(1, 12))

    # 5) Narrative Summary
    story.append(banner("5. Narrative Summary", colors.HexColor("#444444")))
    story.append(Spacer(1, 8))

    strengths = narrative.get("strengths", []) or []
    gaps = narrative.get("gaps", []) or []
    recommendation = narrative.get("recommendation", "")

    story.append(Paragraph("<b>Key Strengths:</b>", styles["BodyX"]))
    for i, s in enumerate(strengths, start=1):
        story.append(Paragraph(f"{i}. {s}", styles["BodyX"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Gaps and Risks:</b>", styles["BodyX"]))
    for i, g in enumerate(gaps, start=1):
        story.append(Paragraph(f"{i}. {g}", styles["BodyX"]))
    story.append(Spacer(1, 8))

    if recommendation:
        story.append(Paragraph(f"<b>Overall Recommendation:</b> {recommendation}", styles["BodyX"]))

    doc.build(story)
    return buffer.getvalue()


def report_data_to_html(report_data: dict) -> str:
    # Create a self-contained, styled HTML report from report_data (safe-escapes dynamic text).
    import html as _html

    if not isinstance(report_data, dict) or not report_data:
        return "<html><body><p>No report_data available.</p></body></html>"

    def esc(x):
        return _html.escape("" if x is None else str(x))

    def badge(text, cls):
        return f'<span class="badge {cls}">{esc(text)}</span>'

    def status_badge(status):
        s = (status or "").upper()
        cls = "neutral"
        if s == "PASS":
            cls = "pass"
        elif s == "FAIL":
            cls = "fail"
        elif s == "REVIEW":
            cls = "review"
        return badge(s, cls)

    decision = (report_data.get("decision") or "REVIEW").upper()
    decision_cls = {"PASS": "pass", "FAIL": "fail", "REVIEW": "review"}.get(decision, "neutral")

    title = report_data.get("title") or f"Candidate Scoring Report – {esc(report_data.get('role_applied',''))}"
    candidate_name = report_data.get("candidate_name", "")
    role_applied = report_data.get("role_applied", "")
    final_score = report_data.get("final_score", "")
    totals = report_data.get("totals", {}) or {}
    narrative = report_data.get("narrative", {}) or {}

    compliance = report_data.get("compliance", []) or []
    must = report_data.get("must_have", {}) or {}
    nice = report_data.get("nice_to_have", {}) or {}

    comp_rows = "\n".join(
        f"<tr><td>{esc(r.get('item',''))}</td><td>{status_badge(r.get('status',''))}</td><td>{esc(r.get('details',''))}</td></tr>"
        for r in compliance
    )

    def req_rows(reqs):
        out = []
        for r in (reqs or []):
            weight = float(r.get("weight_pct", 0) or 0)
            contrib = float(r.get("weighted_contribution_pct", 0) or 0)
            score = int(r.get("score_0_5", 0) or 0)
            out.append(
                "<tr>"
                f"<td class='mono'>{esc(r.get('idx',''))}</td>"
                f"<td>{esc(r.get('label',''))}</td>"
                f"<td class='center'>{esc(f'{weight:.2f}%')}</td>"
                f"<td class='center'>{esc(score)}</td>"
                f"<td class='center'>{esc(f'{contrib:.2f}%')}</td>"
                f"<td class='evidence'>{esc(r.get('evidence',''))}</td>"
                "</tr>"
            )
        return "\n".join(out)

    must_rows = req_rows(must.get("requirements", []))
    nice_rows = req_rows(nice.get("requirements", []))

    strengths = "\n".join(f"<li>{esc(x)}</li>" for x in (narrative.get("strengths", []) or []))
    gaps = "\n".join(f"<li>{esc(x)}</li>" for x in (narrative.get("gaps", []) or []))

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --muted: #aab4d4;
      --text: #e9ecff;
      --line: rgba(255,255,255,.10);
      --shadow: 0 12px 30px rgba(0,0,0,.35);
      --radius: 16px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 18px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: radial-gradient(1000px 600px at 20% 0%, #1a275f 0%, rgba(26,39,95,0) 60%),
                  radial-gradient(1000px 600px at 80% 0%, #2a1658 0%, rgba(42,22,88,0) 60%),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .header {{
      display: flex; gap: 18px; align-items: flex-start; justify-content: space-between;
      padding: 18px 20px;
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow);
    }}
    h1 {{ margin: 0; font-size: 20px; letter-spacing: .2px; }}
    .sub {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .pillrow {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.04);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      display: flex; align-items: center; gap: 8px;
    }}
    .badge {{
      display: inline-flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 12px;
      padding: 4px 10px; border-radius: 999px;
      border: 1px solid rgba(255,255,255,.18);
    }}
    .badge.pass {{ background: rgba(34,197,94,.14); color: #b8ffd0; border-color: rgba(34,197,94,.35); }}
    .badge.fail {{ background: rgba(239,68,68,.14); color: #ffd0d0; border-color: rgba(239,68,68,.35); }}
    .badge.review {{ background: rgba(245,158,11,.16); color: #ffe7b5; border-color: rgba(245,158,11,.40); }}
    .badge.neutral {{ background: rgba(100,116,139,.16); color: #d7deea; border-color: rgba(100,116,139,.40); }}

    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }}
    .card {{
      padding: 16px 16px; background: rgba(255,255,255,.04);
      border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card h2 {{ margin: 0 0 10px 0; font-size: 15px; }}
    .muted {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
      overflow: hidden;
      border-radius: 12px;
    }}
    th, td {{ padding: 10px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{
      text-align: left;
      color: #cfd7ff;
      background: rgba(255,255,255,.05);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .6px;
    }}
    tr:hover td {{ background: rgba(255,255,255,.03); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .center {{ text-align: center; white-space: nowrap; }}
    .evidence {{ color: #dbe2ff; max-width: 520px; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 900px) {{ .split {{ grid-template-columns: 1fr; }} }}
    .callout {{
      padding: 12px 14px; border-radius: 12px;
      background: rgba(255,255,255,.04); border: 1px solid var(--line);
      color: var(--muted);
    }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    li {{ margin: 6px 0; color: #dbe2ff; }}
    footer {{ margin-top: 14px; color: rgba(255,255,255,.45); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1>{esc(title)}</h1>
        <div class="sub">Rendered from structured scoring output (report_data).</div>
        <div class="pillrow">
          <div class="pill"><span class="muted">Candidate</span> <strong>{esc(candidate_name)}</strong></div>
          <div class="pill"><span class="muted">Role</span> <strong>{esc(role_applied)}</strong></div>
          <div class="pill"><span class="muted">Final Score</span> <strong>{esc(final_score)}</strong></div>
          <div class="pill"><span class="muted">Decision</span> {badge(decision, decision_cls)}</div>
        </div>
      </div>
      <div style="min-width: 260px;">
        <div class="callout">
          <div class="muted"><strong>Rationale</strong></div>
          <div style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('rationale',''))}</div>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Compliance (Gating – Not Scored)</h2>
        <table>
          <thead><tr><th>Item</th><th>Status</th><th>Details</th></tr></thead>
          <tbody>{comp_rows}</tbody>
        </table>
      </div>

      <div class="card">
        <h2>Must-have (Total Weight = 90.00%)</h2>
        <div class="muted">Subtotal: <span class="mono">{esc(must.get('subtotal_weighted',''))}</span></div>
        <table>
          <thead>
            <tr><th>#</th><th>Requirement</th><th class="center">Weight</th><th class="center">Score</th><th class="center">Wt Score</th><th>Evidence</th></tr>
          </thead>
          <tbody>{must_rows}</tbody>
        </table>
      </div>

      <div class="card">
        <h2>Nice-to-have (Total Weight = 10.00%)</h2>
        <div class="muted">Subtotal: <span class="mono">{esc(nice.get('subtotal_weighted',''))}</span></div>
        <table>
          <thead>
            <tr><th>#</th><th>Skill</th><th class="center">Weight</th><th class="center">Score</th><th class="center">Wt Score</th><th>Evidence</th></tr>
          </thead>
          <tbody>{nice_rows}</tbody>
        </table>
      </div>

      <div class="card">
        <h2>Totals and Final Decision</h2>
        <div class="split">
          <div class="callout"><div class="muted">Weighted Must-have</div><div class="mono" style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('weighted_must_have',''))}</div></div>
          <div class="callout"><div class="muted">Weighted Nice-to-have</div><div class="mono" style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('weighted_nice_to_have',''))}</div></div>
          <div class="callout"><div class="muted">Total Weighted</div><div class="mono" style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('total_weighted',''))}</div></div>
          <div class="callout"><div class="muted">Floor Rule</div><div style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('floor_rule',''))}</div></div>
          <div class="callout"><div class="muted">Compliance</div><div style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('compliance_summary',''))}</div></div>
          <div class="callout"><div class="muted">Final Decision</div><div style="margin-top:6px; color:#dbe2ff;">{esc(totals.get('final_decision_line',''))}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Narrative Summary</h2>
        <div class="split">
          <div><div class="muted"><strong>Key Strengths</strong></div><ul>{strengths}</ul></div>
          <div><div class="muted"><strong>Gaps and Risks</strong></div><ul>{gaps}</ul></div>
        </div>
        <div class="callout" style="margin-top:12px;">
          <div class="muted"><strong>Recommendation</strong></div>
          <div style="margin-top:6px; color:#dbe2ff;">{esc(narrative.get('recommendation',''))}</div>
        </div>
      </div>
    </div>

    <footer>Generated HTML report. You can download this file and open it in any browser.</footer>
  </div>
</body>
</html>
"""
    return html_doc


def init_state():
    # Rubric tab state
    st.session_state.setdefault("role_rubric", "")
    st.session_state.setdefault("rubric_name", "")
    st.session_state.setdefault("jd_rubric_paste", "")
    st.session_state.setdefault("generated_rubric_yaml", "")

    # Eval tab state
    st.session_state.setdefault("role_eval", "")
    st.session_state.setdefault("jd_eval_paste", "")
    st.session_state.setdefault("preset_rubric_yaml_area", "")
    st.session_state.setdefault("pasted_cvs", "")

    # Results (survive reruns + download clicks)
    st.session_state.setdefault("last_report_data", None)
    st.session_state.setdefault("last_pdf_bytes", b"")
    st.session_state.setdefault("last_html_report", "")


def clear_rubric_form():
    st.session_state["role_rubric"] = ""
    st.session_state["rubric_name"] = ""
    st.session_state["jd_rubric_paste"] = ""
    st.session_state["generated_rubric_yaml"] = ""


def clear_eval_form():
    st.session_state["role_eval"] = ""
    st.session_state["jd_eval_paste"] = ""
    st.session_state["preset_rubric_yaml_area"] = ""
    st.session_state["pasted_cvs"] = ""
    st.session_state["last_report_data"] = None
    st.session_state["last_pdf_bytes"] = b""
    st.session_state["last_html_report"] = ""


init_state()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="JD→CV Scorer (Form)", layout="wide")
st.title("JD→CV Scorer — Form UI (No Chat)")

tab1, tab2 = st.tabs(["1) Generate Rubric", "2) Evaluate CVs"])

# ----------------------------
# TAB 1: Generate Rubric
# ----------------------------
with tab1:
    st.subheader("Generate a preset rubric from a Job Description")

    # Clear button (does not depend on form submission)
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("Clear rubric form"):
            clear_rubric_form()
            st.rerun()

    with st.form("rubric_form", clear_on_submit=False):
        role_applied = st.text_input(
            "Role applied (type manually)",
            placeholder="e.g., Senior Frontend Engineer",
            key="role_rubric",
        )
        rubric_name = st.text_input(
            "Rubric name (optional)",
            placeholder="e.g., FE_Senior_2026_01",
            key="rubric_name",
        )

        jd_file = st.file_uploader(
            "Upload JD (TXT / PDF / DOCX)",
            type=["txt", "pdf", "docx"],
            key="jd_rubric_file",
        )
        jd_text_paste = st.text_area(
            "…or paste JD text",
            height=180,
            key="jd_rubric_paste",
        )

        generate = st.form_submit_button("Generate Rubric")

    if generate:
        jd_text = st.session_state["jd_rubric_paste"].strip()
        if not jd_text and jd_file:
            jd_text = extract_text(jd_file).strip()

        if not st.session_state["role_rubric"].strip():
            st.error("Please type the Role Applied.")
        elif not jd_text:
            st.error("Please provide a Job Description (upload or paste).")
        else:
            if not RUBRIC_TEMPLATE_TEXT.strip():
                st.error("rubric_template.yaml not found in ./knowledge. Please add it (knowledge/rubric_template.yaml).")
            elif not api_key:
                st.error("Missing OPENAI_API_KEY. Add it to your local .env or Streamlit Secrets.")
            else:
                with st.spinner("Generating rubric..."):
                    try:
                        prompt = build_generate_rubric_prompt(
                            role_applied=st.session_state["role_rubric"].strip(),
                            jd_text=jd_text,
                            rubric_name=st.session_state["rubric_name"].strip() or None,
                        )
                        resp = client.responses.create(
                            model=model,
                            input=[{"role": "user", "content": prompt}],
                        )
                        data = {"rubric_yaml": resp.output_text or ""}
                    except Exception as e:
                        st.error(f"OpenAI request failed: {e}")
                        data = None

            if data:
                if "error" in data:
                    st.error(data["error"])
                else:
                    rubric_yaml = data.get("rubric_yaml", "")
                    st.session_state["generated_rubric_yaml"] = rubric_yaml
                    st.session_state["preset_rubric_yaml_area"] = rubric_yaml  # auto-fill eval tab
                    st.success("Rubric generated and saved for evaluation (session).")

    if st.session_state["generated_rubric_yaml"].strip():
        st.text_area("Generated rubric (YAML)", value=st.session_state["generated_rubric_yaml"], height=320)

        st.download_button(
            "Download rubric YAML",
            data=st.session_state["generated_rubric_yaml"].encode("utf-8"),
            file_name="generated_rubric.yaml",
            mime="text/yaml",
        )

# ----------------------------
# TAB 2: Evaluate CVs
# ----------------------------
with tab2:
    st.subheader("Evaluate CVs (uses preset rubric if provided)")

    # Clear button
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("Clear evaluation form"):
            clear_eval_form()
            st.rerun()

    with st.form("eval_form", clear_on_submit=False):
        st.text_input(
            "Role applied (type manually)",
            placeholder="e.g., Senior Frontend Engineer",
            key="role_eval",
        )

        st.markdown("## Job Description")
        jd_file_eval = st.file_uploader(
            "Upload JD (TXT / PDF / DOCX)",
            type=["txt", "pdf", "docx"],
            key="jd_eval_file",
        )
        st.text_area("…or paste JD text", height=160, key="jd_eval_paste")

        st.markdown("## Preset Rubric (optional)")
        st.caption("Generated rubric (tab 1) auto-fills here. You can edit or paste your own.")
        st.text_area("Preset rubric YAML (optional)", height=180, key="preset_rubric_yaml_area")

        st.markdown("## CVs")
        cv_files = st.file_uploader(
            "Upload CV files (TXT / PDF / DOCX) — select multiple",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="cv_files",
        )

        st.text_area(
            "…and/or paste CV text (one or multiple). Separate multiple CVs with a line containing only: ---",
            height=200,
            key="pasted_cvs",
        )

        submitted = st.form_submit_button("Score")

    if submitted:
        role = st.session_state["role_eval"].strip()
        jd_text = st.session_state["jd_eval_paste"].strip()

        if not jd_text and jd_file_eval:
            jd_text = extract_text(jd_file_eval).strip()

        if not role:
            st.error("Please type the Role Applied.")
        elif not jd_text:
            st.error("Please provide a Job Description (upload or paste).")
        else:
            cvs = []

            # Uploaded CVs
            if cv_files:
                for f in cv_files:
                    text = extract_text(f).strip()
                    if text:
                        cvs.append({"name": f.name, "text": text})

            # Pasted CVs
            pasted_parts = split_pasted_cvs(st.session_state["pasted_cvs"])
            for i, part in enumerate(pasted_parts, start=1):
                cvs.append({"name": f"pasted_cv_{i}.txt", "text": part})

            if not cvs:
                st.error("Please upload at least one CV or paste at least one CV.")
            else:
                preset_rubric = st.session_state["preset_rubric_yaml_area"].strip() or None

                if not api_key:
                    st.error("Missing OPENAI_API_KEY. Add it to your local .env or Streamlit Secrets.")
                    data = None
                else:
                    with st.spinner("Scoring..."):
                        try:
                            prompt = build_score_prompt(
                                role_applied=role,
                                jd_text=jd_text,
                                cvs=cvs,
                                preset_rubric_yaml=preset_rubric,
                            )
                            resp = client.responses.create(
                                model=model,
                                input=[{"role": "user", "content": prompt}],
                            )
                            data = {"result": resp.output_text}
                        except Exception as e:
                            st.error(f"OpenAI request failed: {e}")
                            data = None

                if data:
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        raw = data.get("result", "")
                        try:
                            model_obj = json.loads(raw)
                            st.session_state["last_report_data"] = model_obj.get("report_data", None)
                        except Exception:
                            st.error("Model returned non-JSON output. Showing raw output below.")
                            st.code(raw)

# ----- Render results from session_state (survives reruns/downloads) -----
report_data = st.session_state.get("last_report_data")

if report_data:
    st.markdown("## Results")

    # Build HTML report (cached in session state)
    html_report = st.session_state.get("last_html_report") or ""
    if not html_report:
        html_report = report_data_to_html(report_data)
        st.session_state["last_html_report"] = html_report

    # Show HTML report inside the app window
    st.components.v1.html(html_report, height=1100, scrolling=True)

    st.markdown("## Downloads")

    st.download_button(
        "Download Evaluation Report (HTML)",
        data=html_report.encode("utf-8"),
        file_name="evaluation_report.html",
        mime="text/html",
    )

    # Optional: keep PDF download
    pdf_bytes = st.session_state.get("last_pdf_bytes") or b""
    if not pdf_bytes:
        pdf_bytes = pdf_bytes_from_report_data(report_data, markdown_fallback="")
        st.session_state["last_pdf_bytes"] = pdf_bytes

    st.download_button(
        "Download Evaluation Report (PDF)",
        data=pdf_bytes,
        file_name="evaluation_report.pdf",
        mime="application/pdf",
    )

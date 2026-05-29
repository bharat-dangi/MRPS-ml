"""
Generate a synthetic section-classification training set.

Produces 200 labelled text blocks across 5 section types
(Education, Experience, Skills, Summary, Other) using domain-aware templates.
Each generated block contains the kind of phrasing and structure a real-world
resume section would use, randomised for diversity.

Output: ml/data/section_annotations/synthetic_sections.jsonl
Format: {"text": str, "label": str}

Why synthetic: the CSV calls for 200 hand-annotated Kaggle resumes. We don't
ship that data in the repo; instead this script produces a deterministic,
inspectable dataset that the textcat model trains on. Anyone running the
pipeline against real resumes can replace the JSONL with their own labels
without code changes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABELS = ["Education", "Experience", "Skills", "Summary", "Other"]
OUT = Path(__file__).resolve().parent.parent / "data" / "section_annotations" / "synthetic_sections.jsonl"

DEGREES = [
    "Bachelor of Science in Computer Science", "Bachelor of Engineering",
    "Master of Information Technology", "Master of Business Administration",
    "Bachelor of Commerce", "Bachelor of Arts in Psychology",
    "Diploma in Nursing", "Associate Degree in Accounting",
    "PhD in Machine Learning", "Master of Data Science",
    "Bachelor of Pharmacy", "Bachelor of Education",
    "Master of Public Health", "Bachelor of Laws",
]
UNIVERSITIES = [
    "Australian Catholic University", "University of Sydney", "Monash University",
    "RMIT University", "Stanford University", "MIT", "Harvard University",
    "University of Melbourne", "University of New South Wales",
    "Queensland University of Technology", "Deakin University", "University of Cambridge",
]
TITLES_BY_DOMAIN = {
    "tech": ["Software Engineer", "Backend Developer", "Data Scientist", "DevOps Engineer", "Machine Learning Engineer"],
    "finance": ["Financial Analyst", "Senior Accountant", "Investment Analyst", "Treasury Manager"],
    "health": ["Registered Nurse", "Clinical Coordinator", "Medical Coder", "Pharmacist"],
    "marketing": ["Digital Marketing Manager", "SEO Specialist", "Content Strategist", "Brand Manager"],
    "ops": ["Operations Manager", "Supply Chain Analyst", "Procurement Specialist", "Logistics Coordinator"],
    "hr": ["HR Business Partner", "Talent Acquisition Specialist", "HR Generalist"],
}
COMPANIES = [
    "Northwind Group", "Brightpath Solutions", "Vertex Partners", "Summit Co",
    "Clearwater Holdings", "Ironwood Industries", "Blue Harbor Group", "Meridian Partners",
    "Acme Corp", "Globex Inc", "Initech", "Soylent Corp", "Pied Piper",
]
SKILLS_BY_DOMAIN = {
    "tech": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS", "Git", "React", "TypeScript"],
    "finance": ["Financial modeling", "Excel", "SAP", "Power BI", "VBA", "GAAP", "IFRS accounting"],
    "health": ["Patient care", "IV therapy", "EHR", "Epic", "BLS certification", "Phlebotomy", "Triage"],
    "marketing": ["SEO", "Google Analytics", "HubSpot", "Content marketing", "Campaign management", "A/B testing"],
    "ops": ["Supply chain planning", "ERP", "Inventory management", "Procurement", "Lean Six Sigma"],
    "hr": ["Employee relations", "Performance management", "HRIS", "Workday HCM", "Talent acquisition"],
}
SUMMARY_OPENERS = [
    "Detail-oriented professional with",
    "Results-driven specialist with",
    "Experienced practitioner offering",
    "Highly motivated candidate bringing",
    "Versatile professional with a track record of",
    "Strategic operator delivering",
    "Dedicated team player with",
]
OTHER_SNIPPETS = [
    "References available upon request.",
    "LinkedIn: linkedin.com/in/{slug}",
    "Phone: 0400 123 456 | Email: {email}",
    "Hobbies: bushwalking, photography, and amateur radio.",
    "Languages: English (native), Mandarin (fluent), French (conversational).",
    "Volunteering: Foodbank Australia, Lifeline, Habitat for Humanity.",
    "Address: 42 Sample Street, Sydney NSW 2000",
    "Driver's licence: Class C (Full), endorsements: LR, MR.",
    "Available immediately for full-time roles in Sydney or Melbourne.",
    "Citizenship: Australian, with full work rights.",
]


def _education_block(rng: random.Random) -> str:
    n = rng.randint(1, 2)
    lines: list[str] = []
    for _ in range(n):
        deg = rng.choice(DEGREES)
        uni = rng.choice(UNIVERSITIES)
        end_year = rng.randint(2010, 2024)
        start_year = end_year - rng.randint(2, 5)
        gpa = round(rng.uniform(3.2, 4.0), 2)
        lines.append(
            f"{deg} — {uni} ({start_year}–{end_year}). GPA {gpa}/4.0. "
            f"Coursework: {rng.choice(['Data Structures', 'Macroeconomics', 'Anatomy', 'Digital Marketing'])}, "
            f"{rng.choice(['Algorithms', 'Corporate Finance', 'Pharmacology', 'Consumer Behaviour'])}."
        )
    return "\n".join(lines)


def _experience_block(rng: random.Random) -> str:
    domain = rng.choice(list(TITLES_BY_DOMAIN.keys()))
    title = rng.choice(TITLES_BY_DOMAIN[domain])
    company = rng.choice(COMPANIES)
    end_year = rng.randint(2018, 2025)
    start_year = end_year - rng.randint(1, 5)
    bullets = rng.sample(
        [
            "Led cross-functional initiatives across product, design, and engineering teams.",
            "Owned end-to-end delivery of customer-facing features impacting 50k+ users.",
            "Reduced operational costs by 18% through process redesign and automation.",
            "Built dashboards to give stakeholders real-time visibility into KPIs.",
            "Mentored 4 junior team members through structured pairing and code reviews.",
            "Negotiated supplier contracts saving the business $250k annually.",
            "Wrote standard operating procedures adopted across three sites.",
            "Drove a 30% improvement in customer satisfaction over two quarters.",
        ],
        k=rng.randint(3, 5),
    )
    return f"{title} | {company} | {start_year}–{end_year}\n- " + "\n- ".join(bullets)


def _skills_block(rng: random.Random) -> str:
    domain = rng.choice(list(SKILLS_BY_DOMAIN.keys()))
    primary = rng.sample(SKILLS_BY_DOMAIN[domain], k=min(5, len(SKILLS_BY_DOMAIN[domain])))
    extras = ["Stakeholder management", "Communication", "Problem solving", "Agile", "Documentation"]
    extras = rng.sample(extras, k=rng.randint(2, 4))
    layout = rng.choice(["bullets", "csv", "groups"])
    if layout == "bullets":
        return "- " + "\n- ".join(primary + extras)
    if layout == "csv":
        return ", ".join(primary + extras) + "."
    return (
        f"Technical: {', '.join(primary)}.\n"
        f"Soft skills: {', '.join(extras)}."
    )


def _summary_block(rng: random.Random) -> str:
    opener = rng.choice(SUMMARY_OPENERS)
    years = rng.randint(3, 12)
    domain = rng.choice(list(TITLES_BY_DOMAIN.keys()))
    focus = rng.choice(
        ["product growth", "regulatory compliance", "operational efficiency",
         "patient outcomes", "campaign performance", "supply chain resilience",
         "developer productivity"]
    )
    return (
        f"{opener} {years}+ years of experience focused on {focus}. "
        f"Track record of delivering measurable outcomes in {domain} environments, "
        f"partnering closely with stakeholders to translate ambiguous problems into "
        f"actionable strategies. Seeking a role where I can continue to grow and "
        f"contribute to a team committed to excellence."
    )


def _other_block(rng: random.Random) -> str:
    n = rng.randint(2, 4)
    items = rng.sample(OTHER_SNIPPETS, k=n)
    return "\n".join(
        item.format(slug=f"sample-user-{rng.randint(100, 999)}", email=f"user{rng.randint(100, 999)}@example.com")
        for item in items
    )


GENERATORS = {
    "Education": _education_block,
    "Experience": _experience_block,
    "Skills": _skills_block,
    "Summary": _summary_block,
    "Other": _other_block,
}


def generate(n_per_label: int = 40, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    samples: list[dict[str, str]] = []
    for label in LABELS:
        for _ in range(n_per_label):
            samples.append({"label": label, "text": GENERATORS[label](rng)})
    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-label", type=int, default=40,
                        help="Samples per label (200 total = 40 × 5 labels)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = generate(args.per_label, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} samples → {OUT.relative_to(Path.cwd())}")
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    for label, count in sorted(counts.items()):
        print(f"  {label:12s} {count}")


if __name__ == "__main__":
    main()

"""Skill normalisation for fairer matching during scoring.

ALIASES collapse surface variants both ways (`k8s` ↔ `kubernetes`).
HIERARCHY is asymmetric: candidate "PostgreSQL" satisfies JD "SQL", but the
reverse must not auto-match — generic claims don't imply specific expertise.
FAMILIES award partial credit between peer skills (PostgreSQL ↔ MariaDB).
"""
from __future__ import annotations

# Variant → canonical. All keys + values must be lowercase.
ALIASES: dict[str, str] = {
    # databases
    "postgres": "postgresql",
    "psql": "postgresql",
    "pg": "postgresql",
    "mssql": "microsoft sql server",
    "ms sql": "microsoft sql server",
    "ms sql server": "microsoft sql server",
    "sql server": "microsoft sql server",
    "mongo": "mongodb",
    "redis cache": "redis",
    "dynamo": "dynamodb",
    # languages
    "js": "javascript",
    "ecmascript": "javascript",
    "ts": "typescript",
    "py": "python",
    "go": "golang",
    "objective c": "objective-c",
    "c sharp": "c#",
    "csharp": "c#",
    "cpp": "c++",
    "c plus plus": "c++",
    # frameworks
    "node": "node.js",
    "nodejs": "node.js",
    "next": "next.js",
    "nextjs": "next.js",
    "nuxt": "nuxt.js",
    "nuxtjs": "nuxt.js",
    "vuejs": "vue",
    "vue.js": "vue",
    "rails": "ruby on rails",
    "spring": "spring boot",
    "dotnet": ".net",
    "dot net": ".net",
    # cloud / devops
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "gcloud": "google cloud platform",
    "azure": "microsoft azure",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    # ML
    "tf": "tensorflow",
    "tensor flow": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "pt": "pytorch",
    "hf": "huggingface",
    "hugging face": "huggingface",
    "transformers library": "transformers",
    # data tools
    "spark": "apache spark",
    "kafka": "apache kafka",
    "airflow": "apache airflow",
    "es": "elasticsearch",
    "elastic search": "elasticsearch",
    # misc
    "ci/cd": "cicd",
    "ci-cd": "cicd",
    "version control": "git",
    "gitlab ci": "gitlab",
    "github actions": "github",
}

# Specific → broader categories it implies. Only add a parent when knowing
# the specific genuinely demonstrates the parent (PostgreSQL → SQL is safe;
# FastAPI → REST is not, since a FastAPI dev hasn't necessarily designed APIs).
HIERARCHY: dict[str, list[str]] = {
    # relational DBs all imply SQL
    "postgresql": ["sql", "relational database"],
    "mysql": ["sql", "relational database"],
    "sqlite": ["sql", "relational database"],
    "mariadb": ["sql", "relational database"],
    "microsoft sql server": ["sql", "relational database"],
    "oracle": ["sql", "relational database"],
    "oracle database": ["sql", "relational database"],
    "db2": ["sql", "relational database"],
    # nosql family
    "mongodb": ["nosql"],
    "dynamodb": ["nosql"],
    "redis": ["nosql"],
    "cassandra": ["nosql"],
    "couchdb": ["nosql"],
    # web frameworks → host language
    "fastapi": ["python"],
    "django": ["python"],
    "flask": ["python"],
    "spring boot": ["java"],
    "express": ["javascript", "node.js"],
    "rails": ["ruby"],
    "laravel": ["php"],
    # frontend → host language
    "react": ["javascript"],
    "vue": ["javascript"],
    "angular": ["typescript", "javascript"],
    "next.js": ["react", "javascript"],
    "nuxt.js": ["vue", "javascript"],
    "svelte": ["javascript"],
    # ML libs → python + ML
    "tensorflow": ["python", "machine learning"],
    "pytorch": ["python", "machine learning"],
    "scikit-learn": ["python", "machine learning"],
    "keras": ["python", "machine learning"],
    "xgboost": ["python", "machine learning"],
    "lightgbm": ["python", "machine learning"],
    # data libs → python
    "pandas": ["python"],
    "numpy": ["python"],
    "matplotlib": ["python"],
    "seaborn": ["python"],
    # container / orchestration
    "kubernetes": ["container orchestration", "devops"],
    "docker": ["containers", "devops"],
    "openshift": ["kubernetes", "container orchestration", "devops"],
    # cloud → cloud category
    "amazon web services": ["cloud"],
    "google cloud platform": ["cloud"],
    "microsoft azure": ["cloud"],
    # data tools
    "apache spark": ["big data", "distributed computing"],
    "apache kafka": ["streaming", "messaging"],
    "apache airflow": ["data pipelines", "orchestration"],
    "elasticsearch": ["search"],
    # mobile
    "swift": ["ios"],
    "kotlin": ["android"],
    "react native": ["javascript", "mobile"],
    "flutter": ["dart", "mobile"],
}


# Narrow "interchangeable" families — peer skills where transfer is genuinely
# easy (recruiter would happily count one for the other). Keep tight; don't add
# pairs like {Django, FastAPI} where API conventions differ.
FAMILIES: dict[str, frozenset[str]] = {
    # Relational databases — SQL syntax + relational modelling transfers
    "rdbms": frozenset({
        "postgresql", "mysql", "sqlite", "mariadb",
        "microsoft sql server", "oracle", "oracle database", "db2",
    }),
    # Major cloud providers — compute/storage/IAM concepts transfer
    "cloud_providers": frozenset({
        "amazon web services", "google cloud platform", "microsoft azure",
    }),
    # Container orchestration platforms — same Kubernetes API surface
    "container_orchestration": frozenset({"kubernetes", "openshift"}),
}

# Awarded for sibling-family or generic-implies-specific matches — transferable
# foundation but not full credit.
PARTIAL_CREDIT = 0.5


def _family_of(skill: str) -> frozenset[str] | None:
    """Return the FAMILIES set containing `skill`, or None if it isn't in one."""
    for family in FAMILIES.values():
        if skill in family:
            return family
    return None


def canonical(skill: str) -> str:
    """Lowercase + alias-collapse a single skill string."""
    s = skill.strip().lower()
    return ALIASES.get(s, s)


def normalize(skills: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    """Apply `canonical` to every skill and drop empties."""
    return {canonical(s) for s in skills if s and s.strip()}


def expand_candidate(skills: list[str] | set[str]) -> set[str]:
    """Normalise + expand a candidate's skills through HIERARCHY (one level, no recursion)."""
    base = normalize(skills)
    expanded = set(base)
    for s in base:
        expanded.update(HIERARCHY.get(s, []))
    return expanded

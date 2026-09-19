# AI Model Risk & Assurance Copilot

## 1. Project Identity

Project name:

**AI Model Risk & Assurance Copilot for RBI Compliance**

We are a team of five beginner developers building a one-month MVP.

The MVP focuses **only on credit-scoring models**.

The purpose of the system is to provide an AI-assisted model risk and assurance workflow that combines technical model analysis with RBI compliance evidence.

The system is intended to eventually:

1. Load a credit-scoring dataset/model.
2. Perform model analysis.
3. Generate SHAP/LIME explainability.
4. Perform fairness analysis.
5. Perform drift detection.
6. Map technical findings to selected RBI requirements.
7. Retrieve relevant RBI source material using RAG.
8. Use an LLM to generate an evidence-grounded compliance report.
9. Display results through a dashboard.

Do not expand the MVP beyond credit-scoring models unless the team explicitly approves it.


## 2. Team

There are five team members.

### Namitha — Model / Data

Owns:

- Credit model
- Dataset preprocessing
- Prediction
- Model training
- Model evaluation
- Model saving/loading
- Model prediction interface

### Manas — Explainability

Owns:

- SHAP
- LIME
- Explainability outputs
- Explainability interface

### Arushi — Fairness / Drift

Owns:

- Fairlearn
- Demographic parity
- Disparate impact
- PSI
- KS
- Drift analysis
- Fairness/drift interfaces

### Nidhi — RBI Compliance / Rule Engine / RAG

Owns:

- RBI rule repository
- RBI clause structure
- Rule engine
- RBI metadata
- Compliance mapping
- RAG
- RBI source retrieval

### Khushi — API / Dashboard / Integration

Owns:

- FastAPI
- Streamlit dashboard
- Mock data display
- API integration
- Overall system integration

### Ownership Rule

During independent development, each member owns their assigned module.

Do not unnecessarily modify another member's module.

If a change affects another member's module or interface:

1. Identify the affected module.
2. Explain the impact.
3. Do not silently change the interface.
4. Obtain team approval when the change is significant.


## 3. Planned Technology Stack

The planned stack is:

- Python
- FastAPI
- Streamlit
- Pandas
- NumPy
- scikit-learn / XGBoost
- SHAP
- LIME
- Fairlearn
- PSI / KS / Evidently
- LangChain
- ChromaDB or FAISS
- LLM API
- Git / GitHub

These are planned technologies, not requirements to use every library immediately.

Do not introduce unnecessary technologies.

Do not add microservices, cloud infrastructure, databases, frameworks, or other major technologies unless there is a clear requirement and the team approves the decision.

Prefer simple solutions that beginner developers can understand and maintain.


## 4. Development Philosophy

The project uses a **phase-gated development system**.

The team must complete and review one phase before officially beginning the next phase.

Do not implement future-phase functionality early unless the team explicitly requests it.

The project should be developed as a modular system with clear interfaces between modules.

During parallel development, modules must be able to use mock inputs where necessary.

The goal is to prevent one team member from becoming blocked by another team member's unfinished work.


## 5. Development Phases

## Phase 0 — Foundation

### Goal

Build the project skeleton and development foundation.

Do NOT implement sophisticated ML functionality during Phase 0.

Expected structure:

```text
app/
├── models/
├── explainability/
├── fairness/
├── drift/
├── rbi/
├── rag/
├── compliance/
└── api/

dashboard/
data/
tests/
docs/

CLAUDE.md
README.md
requirements.txt

The exact internal file structure may be refined after the architecture is reviewed.

Phase 0 should establish:
Project structure
Python environment/setup
Dependency management
Basic module folders
Test structure
Sample/mock data structure
RBI documentation structure
API skeleton
Dashboard skeleton
Basic documentation
Basic runnable application
Do NOT implement during Phase 0:
Production credit-scoring model
SHAP analysis
LIME analysis
Fairness calculations
Drift calculations
Full RBI rule engine
RAG pipeline
LLM compliance report generation
Full dashboard analytics
Phase 0 checkpoint

Phase 0 is complete only when:

Required folders exist.
Dependencies can be installed.
Python environment works.
Basic imports work.
Tests can run.
API skeleton runs.
Dashboard skeleton runs.
Documentation structure exists.
All five team members can clone/pull the repository.
All five team members can run the basic project.
No unexplained architectural differences exist.

The team must review and approve the checkpoint before moving to Phase 1.

Phase 1 — Independent Module Development

During Phase 1, all five members work in parallel.

Each member works primarily within their assigned module.

Mock inputs are encouraged when the real upstream module is unavailable.

Namitha

Implement:

Dataset preprocessing
Credit model
Training
Evaluation
Prediction
Model saving/loading
Model interface
Manas

Implement:

SHAP
LIME
Explainability output
Explainability interface

Dummy models may be used until Namitha's real model is available.

Arushi

Implement:

Fairlearn analysis
Demographic parity
Disparate impact
PSI
KS
Drift analysis
Fairness/drift interfaces

Sample predictions and sample datasets may be used.

Nidhi

Implement:

RBI rule repository
RBI clause structure
Rule representation
Rule engine foundation
RBI metadata
Compliance mapping foundation

Use sample rules where necessary.

Full RAG implementation belongs primarily to Phase 3 unless explicitly approved earlier.

Khushi

Implement:

FastAPI foundation
API endpoints
Streamlit foundation
Mock data display
API interfaces
Integration foundation

Use mock JSON/results where necessary.

### Phase 1 Implementation Constraints — Nidhi and Khushi

The following constraints are mandatory for Phase 1 implementation and must remain compatible with the interfaces established by the team.

#### Nidhi — RBI Compliance / Rule Engine / RAG

Nidhi must treat analytical findings produced by the fairness and drift modules as authoritative technical outputs.

Nidhi must not:

* Recalculate fairness or drift metrics independently.
* Introduce alternative fairness or drift thresholds.
* Create KS or demographic-parity pass/fail thresholds.
* Modify, reinterpret, or override `PASS`, `WARNING`, `FAIL`, or `PENDING` status values produced by the analytical modules.
* Interpret Attribute 9 (`personal_status_and_sex`) as a standalone `gender` or `sex` field.
* Introduce a codebook or semantic remapping of Attribute 9 categories unless explicitly approved by the team.
* Present synthetic drift scenarios as observed real-world population drift.
* Treat analytical thresholds as RBI regulatory requirements.

Nidhi may:

* Map analytical findings to RBI rules and compliance concepts.
* Attach RBI clauses, metadata, and evidence to technical findings.
* Use the existing analytical metric values and status values as inputs to compliance mapping.
* Use sample RBI rules during Phase 1 where necessary.
* Build the RBI rule repository, clause structure, metadata, and compliance mapping foundation defined in the Phase 1 scope.

The authoritative analytical threshold definitions are maintained in `app/config/thresholds.py` and documented in `docs/thresholds.md`.

Full RAG implementation remains primarily a Phase 3 responsibility unless explicitly approved earlier.

#### Khushi — API / Dashboard / Integration

Khushi must consume the existing analytical module interfaces rather than reimplementing their calculations.

Khushi must not:

* Recalculate fairness or drift metrics in the API layer.
* Recalculate or override PSI, KS, demographic parity, or disparate impact values in the dashboard.
* Introduce separate threshold constants for fairness or drift.
* Change, reinterpret, or override analytical status values.
* Replace calculated analytical results with fabricated values.
* Rename `personal_status_and_sex` to `gender` or `sex`.
* Present synthetic drift data as observed real-world population drift.
* Treat analytical thresholds as RBI regulatory requirements.

Khushi must:

* Preserve the existing fairness and drift output schemas.
* Pass `favorable_label` through to `fairness_report` when specified by the API caller.
* Preserve `PASS`, `WARNING`, `FAIL`, and `PENDING` status values.
* Display calculated analytical results rather than modifying them.
* Use mock data only where the Phase 1 interface explicitly permits it.
* Keep API and dashboard integration compatible with the existing module interfaces.
* Connect the analytical modules through FastAPI and display their results through the dashboard.

The API and dashboard layers are integration and presentation layers. Technical metric calculation remains the responsibility of the analytical modules.



Phase 1 Rule

No team member should have to wait for another member to finish Phase 1.

Every module must have a clearly defined input/output interface.

Phase 1 Checkpoint

Each team member must demonstrate their module independently.

Required:

Namitha — complete
Manas — complete
Arushi — complete
Nidhi — complete
Khushi — complete

The team reviews:

Functionality
Tests
Interfaces
Documentation
Scope
Code quality
Compatibility with the architecture

Only after the five modules pass the checkpoint should their work be merged into main.

Phase 2 — Integration

Phase 2 connects the independent modules.

The intended high-level flow is:

                    Credit Model
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
      Explainability  Fairness     Drift
             |           |           |
             +-----------+-----------+
                         |
                         v
                    RBI Rules
                         |
                         v
                  API / Dashboard

The exact implementation must follow the agreed module interfaces.

Namitha

Connect the real credit model to the pipeline and expose required model outputs.

Manas

Connect SHAP/LIME to the agreed model interface.

Arushi

Connect fairness and drift analysis to the agreed model outputs and datasets.

Nidhi

Connect technical findings to the RBI rule engine and compliance mapping.

Khushi

Connect the modules through FastAPI and display results through the dashboard.

The system should eventually support a command similar to:

python run_assurance.py

with results conceptually similar to:

Model: PASS
Explainability: PASS
Fairness: WARNING
Drift: FAIL
RBI mapping: COMPLETE

These are illustrative outputs only.

Actual results must come from the implemented system.

Phase 3 — RAG + LLM

Phase 3 introduces RAG and LLM functionality.

Expected conceptual flow:

RBI Documents
      |
      v
Document Processing
      |
      v
Chunking
      |
      v
Embeddings
      |
      v
Vector Database
      |
      v
Retrieval
      |
      v
Relevant RBI Evidence
      |
      v
LLM
      |
      v
Evidence-Grounded Report
Namitha

Provide model metadata and technical model results.

Manas

Provide explainability evidence.

Arushi

Provide fairness and drift evidence.

Nidhi

Own:

RBI document processing
Chunking
Embeddings
Vector database
Retrieval
RBI evidence mapping
Khushi

Connect report generation to the API and dashboard.

LLM Rule

The LLM must NOT independently calculate technical metrics.

Python must calculate:

Model metrics
Fairness metrics
Drift metrics
Explainability results

The LLM may:

Explain results
Summarize results
Connect findings to retrieved evidence
Generate recommendations based on supported evidence

The LLM must not invent technical results.

Phase 4 — Dashboard + UX

Everyone contributes to the final dashboard.

Namitha
Model result visualization
Model performance information
Manas
SHAP visualization
LIME/explainability presentation
Arushi
Fairness charts
Drift charts
Nidhi
RBI compliance display
Compliance evidence
Report information
Khushi
Overall dashboard
API integration
UX flow

The dashboard should display calculated results rather than modify or fabricate them.

Phase 5 — Testing + Demo

Phase 5 is a team-wide phase.

Everyone participates in:

Testing
Debugging
Integration
Documentation
Presentation
Demo preparation

Use cross-testing where practical.

Suggested cycle:

Namitha tests Manas
Manas tests Arushi
Arushi tests Nidhi
Nidhi tests Khushi
Khushi tests Namitha

The purpose is to ensure that team members understand and validate code outside their own module.

6. Module Independence

During independent development, mock inputs may be used.

Examples:

Explainability can use a dummy model.
Fairness/drift can use sample predictions.
RBI compliance can use sample rules.
API/dashboard can use mock JSON.

Mock data must be clearly identified as mock data.

Mock results must never be presented as real regulatory evidence or production results.

7. Module Interfaces

Every module must have a clearly defined input/output interface.

Before changing an interface:

Inspect the current interface.
Read relevant documentation.
Identify affected modules.
Explain the proposed change.
Update relevant tests/documentation.
Obtain approval if it is a breaking change.

Do not silently break another module.

Prefer simple interfaces that beginners can understand.

Use simple Python objects, dictionaries, typed structures, or clearly documented schemas where appropriate.

Do not introduce unnecessary abstraction layers.

8. Git / GitHub Workflow

GitHub is the shared source of truth.

Each team member works on a new, short-lived branch per task, not one
permanent branch per person.

Branch naming: feature/<name>-<short-task>

Examples:

feature/namitha-model-foundation
feature/manas-explainability-foundation
feature/arushi-fairness-drift-foundation
feature/nidhi-rbi-foundation
feature/khushi-api-foundation

Do not directly develop feature work on main.

Do not reuse one long-lived branch per person across multiple tasks —
create a new branch for each task and delete it after merge.

See docs/git-workflow.md and docs/team-workflow.md for the full
step-by-step workflow.

Before starting work

Synchronize with the team's agreed base branch, then create a new task
branch from it.

Example:

git checkout main
git pull
git checkout -b feature/yourname-short-task

Before committing
Review the changes.
Run relevant tests/checks.
Confirm unrelated files were not modified.
Confirm the work belongs to the assigned task.
Use a clear commit message.

Example:

git add .
git commit -m "feat: add explainability foundation"
git push
Pull Requests

Work should be merged through a Pull Request whenever practical.

Before merging:

Relevant tests pass.
The task is within scope.
Another team member has reviewed the change.
Cross-module impacts have been considered.

Do not merge unfinished experimental work into main.

Merge conflicts

If a conflict affects another person's module:

Do not blindly choose one side.
Inspect both changes.
Explain the conflict.
Involve the affected team member when necessary.
9. Claude Code Role

Claude Code is the project's primary architect, planner, reviewer, debugging assistant, and project-level reasoning tool.

Claude Code should be used for:

Understanding the entire repository
Architecture planning
Phase planning
Task decomposition
Module interface design
Cross-module reasoning
Implementation planning
Code review
Debugging difficult issues
Integration planning
Documentation
Test review
Git assistance

Claude Code should understand the whole project rather than focusing only on one file.

Before significant changes

Claude Code should:

Read CLAUDE.md.
Read relevant documentation.
Inspect relevant code.
Inspect relevant tests.
Identify the current phase.
Identify the assigned task.
Explain the proposed change.
Identify files that will be affected.
Identify possible cross-module effects.

For small, clearly scoped changes, Claude Code may proceed after explaining the plan.

For major architectural changes, breaking interface changes, or changes affecting another person's module, Claude Code must stop and request human/team approval.

10. Antigravity Role

Antigravity is the primary implementation environment/agent used by the team.

Antigravity should be used for:

Creating files
Implementing assigned functionality
Writing tests
Running tests
Running development commands
Fixing scoped implementation issues
Refactoring within an assigned module

Antigravity should follow the architecture and implementation plan agreed by the team and Claude Code.

Antigravity must not independently redesign the project architecture.

If Antigravity believes a major architectural change is necessary:

Stop the implementation.
Explain why the change is necessary.
Identify affected modules/files.
Bring the issue to Claude Code and the team.
Wait for approval before implementing the architectural change.
11. Claude Code + Antigravity Workflow

The intended workflow is:

Team
 |
 v
Claude Code
 |
 | architecture / planning / task definition
 v
Implementation Brief
 |
 v
Antigravity
 |
 | implementation / tests
 v
Git Branch
 |
 v
Claude Code Review
 |
 v
Human Review
 |
 v
Pull Request
 |
 v
GitHub main

Claude Code should not be treated merely as a prompt generator.

Claude Code should understand and review the actual repository.

Antigravity should not be treated as the project's architectural decision-maker.

The team remains responsible for final decisions.

12. AI Agent Scope Rules

Before making changes, AI agents must determine:

Current project phase.
Assigned team member.
Assigned module.
Exact task.
Relevant files.
Files that must not be changed.

AI agents must:

Read CLAUDE.md.
Read relevant documentation.
Inspect existing code before modifying it.
Respect module ownership.
Follow existing interfaces.
Avoid future-phase functionality.
Avoid unnecessary refactoring.
Avoid unrelated file changes.
Write/update relevant tests.
Run relevant checks.
Report what was changed.

AI agents must not silently:

Change architecture.
Change module interfaces.
Replace the technology stack.
Add unnecessary dependencies.
Modify another member's module.
Implement future phases.
Invent RBI requirements or citations.
13. AI Implementation Briefs

When Claude Code generates a task for Antigravity, the task should specify:

Team member responsible.
Current phase.
Exact objective.
Files that may be modified.
Files/modules that must not be modified.
Existing interfaces to follow.
Implementation requirements.
Tests required.
Acceptance criteria.
Expected final report.

Antigravity should explain its implementation plan before making significant changes.

If the task is ambiguous, the agent should ask for clarification rather than inventing requirements.

14. Human Review

AI-generated code must be reviewed by the responsible team member.

The team should understand what is being committed.

AI-generated code is not automatically correct.

The team is responsible for:

Architectural decisions
Regulatory interpretation
Code approval
Testing
Final implementation
Project presentation
15. Beginner-Friendly Rules

The team members are beginners with limited experience in:

Git
Software architecture
Full-stack ML applications
AI-assisted development

Therefore, AI agents should:

Explain important concepts before asking developers to use them.
Give commands explicitly when necessary.
Explain why changes are being made.
Avoid assuming advanced knowledge.
Explain errors in beginner-friendly language.
Prefer simple solutions.
Avoid unnecessary abstractions.
Explain trade-offs for major decisions.

Do not optimize for complexity.

Optimize for:

Understanding
Maintainability
Correctness
Clear interfaces
Demonstrability
16. Testing Rules

Every meaningful module should have relevant tests.

Tests should verify:

Expected inputs.
Expected outputs.
Error handling where appropriate.
Module interfaces.
Important edge cases.

Do not claim that a feature works without running the relevant checks.

A task is not complete simply because the code was generated.

A task is complete when the implementation, tests, and acceptance criteria have been reviewed.

17. Documentation Rules

Important project decisions should be documented.

Relevant documentation may include:

docs/
├── architecture.md
├── development-phases.md
├── module-interfaces.md
├── thresholds.md
├── git-workflow.md
└── decisions.md

These documents should be created/refined during Phase 0.

Documentation must reflect the actual state of the repository.

Do not document future functionality as already implemented.

Do not create unnecessary documentation for trivial changes.

18. Source of Truth

When instructions conflict, use this priority:

Explicit team-approved decisions.
Current CLAUDE.md.
Current project documentation.
Current repository implementation.
Task-specific instructions.
AI-generated suggestions.

If an AI agent detects a conflict:

Do not silently choose an interpretation.
Explain the conflict.
Identify affected files/modules.
Ask for clarification when necessary.
19. Definition of Done

A task is considered complete only when:

The assigned functionality is implemented.
The implementation stays within the assigned scope.
Relevant tests/checks exist and pass.
Interfaces are respected.
Documentation is updated where necessary.
No unrelated modules were unnecessarily modified.
The responsible developer understands the changes.
Integration requirements are communicated.
The repository remains runnable.

A phase is complete only when its checkpoint criteria have been satisfied and the team has reviewed the result.

20. Current Project Status

Current phase:

PHASE 4 — DASHBOARD + UX (implemented; approved by Manas 2026-09-13)

Phase 0 (Foundation) was completed and signed off on 2026-08-27, with two
accepted limitations that must be closed early in Phase 1. See
docs/decisions.md, "Phase 0 checkpoint sign-off and Phase 1 start".

Phase 1 (Independent Module Development) was completed and signed off on
2026-09-05. See docs/decisions.md, "Phase 1 checkpoint sign-off and Phase 2
start".

Phase 2 (Cross-Module Integration) was completed and signed off on
2026-09-07. See docs/decisions.md, "Phase 2 checkpoint sign-off and Phase 3
start".

The Phase 1 and Phase 2 module contracts remain authoritative unless
explicitly superseded by a later approved decision in docs/decisions.md.

Phase 3 (RAG + LLM) was completed and signed off on 2026-09-11. See
docs/decisions.md, "Phase 3 checkpoint sign-off".

The current objective is Phase 4: present every analytical result the system
already calculates, each visual sourced from the module that owns the
calculation, with no visual modifying or fabricating a value, and with mock,
synthetic, observed, and unverified-evidence states distinguishable on
screen.

The Phase 4 implementation has been merged into main and the checkpoint is
APPROVED BY MANAS for this project review; see docs/decisions.md, "Phase 4
checkpoint sign-off", including "Level of approval" for exactly what that
covers.

NOTE (2026-09-19): the Phase-4 framing below is historical. Phase 5
(model-agnostic assurance), the monitoring lane, and the React frontend have
all since been delivered and merged. See the CURRENT POSITION block later in
this section, plus docs/regulatory-grounding.md and docs/bank-integration.md,
for the verified present state.

Now in scope — Phase 4:

- Domain-owned dashboard panels under dashboard/panels/, one per analytical
  module, with dashboard/dashboard_app.py as a shell that fetches and
  delegates.
- Model result and model performance visualization.
- SHAP/LIME global and per-instance visualization.
- Fairness per-group and drift per-feature charts.
- RBI compliance findings, evidence, and report display with the three
  report layers kept separate.
- The two live walkthroughs (API reachable and API stopped) required by the
  Phase 4 exit criteria. Both were performed by Manas on 2026-09-13.

Phase 4 Definition of Done status: 22 of 22 PASS. The filled checklist and
per-item evidence are in docs/phase4-allocation.md section 10. Both live
walkthroughs (API reachable and API stopped) were performed by Manas on
2026-09-13. The four other owners did not each personally review their own
panel; the approval recorded is Manas's, at the level he authorised. See
docs/decisions.md, "Phase 4 checkpoint sign-off", under "Level of approval".

Evidence coverage limitation — do not overstate it:

The approved RBI corpus currently contains a single 2014 excerpt
(is_excerpt: True, is_current: False). Real retrieval therefore grounds
1 of the 5 report sections; the other 4 return NOT_FOUND. NOT_FOUND means
no verified evidence was retrieved from the indexed corpus — it does NOT
mean that no RBI rule exists. Phase 3 completion must never be described as
complete regulatory coverage.

Carried forward and NOT yet implemented — see docs/decisions.md, "Phase 3
checkpoint sign-off" and "Phase 4 checkpoint sign-off":

Instance-level evidence in the LLM prompt (no approved sampling policy yet)
A drift evidence producer
Live Groq verification in CI
A broader approved RBI corpus
instance_id on the GET /explainability response (the explainability panel
  labels records by position and says so on screen)

SUPERSEDED — the paragraph below described the position before Phase 5. It is
retained only so the change is visible, and must not be read as current:

  "Any external-bank / model-adapter architecture ... the system currently
  supports exactly one model ... Arbitrary external models are NOT supported."

CURRENT POSITION (verified 2026-09-19):

Phase 5 was delivered. The model-adapter architecture EXISTS and three models
are registered and routable end to end through every analytical endpoint:

  german-credit-logistic-regression   scikit-learn, in-process
  german-credit-random-forest         scikit-learn, in-process
  synthetic-bank-credit-v1            XGBoost, served over HTTP via RESTAdapter

Each adapter declares its own model_id, model_version, feature_names,
capabilities, protected_attribute, trained_on and label_semantics. None of
these is inferred. Monitoring, prediction drift, and a monitoring API and
dashboard were delivered after Phase 4 as well.

Still genuinely out of scope, and not implemented:

Authentication and authorization
Persistence / audit storage
Production deployment artifacts
Employee-level compliance (no user or actor concept exists anywhere)
Transaction-level compliance determination
A broader RBI corpus — see docs/regulatory-grounding.md for the counted
  position (19 declared, 0 present, 1 indexed historical document)
The LLM provider decision. Report generation calls Groq and falls back to a
  clearly labelled mock when GROQ_API_KEY is absent or generation fails.

Analytical thresholds (fairness, drift) have a single authoritative
location and must not be independently defined per module. See
docs/thresholds.md. Those thresholds are project/industry conventions,
NOT RBI requirements, and must never be presented as RBI requirements
without a cited RBI source.

Phase 4 is closed for the purposes of this project review. Keep Phase 5
functionality out of scope until the team explicitly authorizes it; nothing
in Phase 5 is designed or implemented.

21. Default Behavior for Claude Code

When Claude Code is started in this repository:

Read CLAUDE.md.
Determine the current phase.
Inspect the repository before making assumptions.
Identify the user's assigned team member/task when provided.
Follow the ownership boundaries.
Explain plans before significant changes.
Keep changes scoped.
Preserve module interfaces.
Test changes.
Report what changed.
Flag architectural conflicts instead of silently resolving them.
Never implement future phases without explicit instruction.

When the user asks for planning, do not automatically implement.

When the user explicitly asks for implementation, inspect the repository first and implement only the requested scope.
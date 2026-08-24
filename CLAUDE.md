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

Each team member works on their own branch.

Recommended branch names:

feature/namitha
feature/manas
feature/arushi
feature/nidhi
feature/khushi

Do not directly develop feature work on main.

Before starting work

Synchronize with the team's agreed base branch.

Example:

git checkout main
git pull

Then switch to the feature branch.

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

PHASE 0 — FOUNDATION

The repository may currently contain only basic files.

Do not assume that future functionality exists.

Do not implement sophisticated:

Credit-scoring models
SHAP
LIME
Fairness analysis
Drift analysis
RBI rule engine
RAG
LLM compliance reports
Dashboard analytics

unless the team explicitly begins the relevant phase.

The immediate goal is to establish the project foundation, architecture documentation, module structure, interfaces, testing structure, development workflow, and basic runnable skeleton.

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
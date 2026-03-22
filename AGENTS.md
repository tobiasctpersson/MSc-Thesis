# AGENTS.md

This repository contains code and artifacts for the MSc thesis work of Tobias Persson and Anton Johansson.

## Repository Overview

- Primary code lives in `Python Code/`.
- Most modeling and analysis work is notebook-based (`.ipynb` files).
- The main top-level supporting folders include:
  - `Cleaned Data/`
  - `Cleaned Data Sets/`
  - `Figures/`
  - `M-ALL Feature Set/`
  - `Papers/`
  - `Raw Data from Tobias Sichert/`

## Key Working Assumptions

- Prefer understanding the relevant notebook or script before editing anything.
- Treat datasets, generated outputs, and thesis artifacts as important research assets: avoid destructive operations.
- The repository may mix source inputs, cleaned data, figures, and experiments, so check paths carefully before changing code.
- Keep changes narrowly scoped to the task at hand.

## Editing Guidance

- Favor minimal, targeted edits over broad refactors.
- Preserve the current project structure unless the user asks for reorganization.
- If converting notebook logic into scripts or utilities, keep behavior aligned with the existing notebook workflow.
- When editing notebooks, be careful not to overwrite important outputs unless that is part of the task.
- Avoid introducing unnecessary dependencies.

## Data and Reproducibility

- Assume data files may be large, local, and not easily reproducible from scratch.
- Do not delete, rename, or move data files without explicit user approval.
- When adding code, prefer clear paths and reproducible loading steps.
- If a result depends on a local file that may not exist elsewhere, call that out clearly.

## Validation

- For Python changes, run the smallest reasonable verification step available.
- If full execution is expensive or blocked by missing local data, explain what was not validated.
- For notebook-heavy changes, prefer validating the affected cells or the extracted logic rather than rerunning every notebook.

## Collaboration Notes

- Git working tree may contain user work; never revert unrelated changes.
- If unexpected file changes appear, treat them as user-owned unless proven otherwise.
- Ask before making high-impact changes to data layout, experiment outputs, or thesis artifacts.

## Good First Places to Look

- `README.md` for the minimal project description.
- `Python Code/` for notebooks and scripts related to data cleaning and modeling.
- `Python Code/ClaudeRandomForest.py` for a standalone Python script entry point.

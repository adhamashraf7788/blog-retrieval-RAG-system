# CONTRIBUTING.md

## GitHub Workflow

We use a simple workflow:

```text
Main Task Branch → Issues → Implementation → Testing → Pull Request → Review → main
```

The goal is to allow each sub-team to work independently while keeping their work compatible and integratable with the rest of the project.

### 1. Main Task / Sub-Team Branches

Each sub-team working on a main task or workstream works on one dedicated branch.

The branch represents the main task/workstream, not an individual feature or Issue.

For example:

- chunking
- data-cleaning-gathering
- dist-training
- emb-vector_db
- model-training
- query-ret-optimization
- spark

**Branch format:**

```
feature/<main-task>
```

Examples:

- `feature/chunking`
- `feature/data-cleaning`
- `feature/emb-vector-db`
- `feature/model-training`
- `feature/query-ret-optimization`

**Issues within the branch**

Each sub-team creates GitHub Issues for the individual features, tasks, or improvements they implement within their main task.

For example:

Branch: `feature/emb-vector-db`

Issues:
- [Emb-Vector DB] Setup vector database
- [Emb-Vector DB] Implement embedding pipeline
- [Emb-Vector DB] Add document indexing
- [Emb-Vector DB] Implement similarity search

Issues should be linked to the relevant branch and used to track the work being implemented.

### 2. Issues

Every significant feature, task, or improvement should have a GitHub Issue.

**Issue format:**

```
[Main Task] Short description
```

Examples:

- [Chunking] Implement semantic chunking
- [Model-Training] Add training pipeline
- [Emb-Vector DB] Implement vector indexing
- [Query-Ret] Add query decomposition

Each Issue should contain, where applicable:

- Short description
- Requirements
- Acceptance criteria
- Dependencies
- Relevant documentation
- Relevant API contract or mock-data references

Issues make the work traceable and show what each sub-team is implementing.

> 📌 See [Example Issue #2](https://github.com/adhamashraf7788/blog-retrieval-RAG-system/issues/2) for reference.

### 3. main and Pull Requests

`main` is the shared, stable branch.

`main` must only be updated through Pull Requests.

- ❌ No direct pushes
- ❌ No force pushes
- ✅ Pull Request required
- ✅ Team review required
- ✅ Tests required

A sub-team should not merge unfinished work into `main`.

A sub-team should open a PR only when its current main task/workstream is in a fully tested and integratable state.

Before opening a PR, make sure:

- The implemented work is complete for the current scope
- Tests have been performed
- The output is working as expected
- Integration with dependent components has been tested where applicable
- Required documentation is updated
- API contracts are updated where applicable
- Mock data is updated where applicable
- No unrelated changes are included

The goal is that anything merged into `main` is a working version that the rest of the project can build on.

### 4. Pull Requests

Every completed main task/workstream is merged into `main` through a Pull Request.

PRs must:

- Link the relevant Issues
- Explain what was implemented
- Mention testing performed
- Mention integration testing where applicable
- Identify important API, schema, or output changes
- Have at least one team-member review
- Contain no unrelated changes

Example:

```
[Emb-Vector DB] Complete vector indexing and retrieval
```

The PR should reference the Issues completed as part of that workstream.

After the PR is merged, the branch can be deleted if the sub-team has finished that workstream.

> 📌 See [Example Pull Request #3](https://github.com/adhamashraf7788/blog-retrieval-RAG-system/pull/3) for reference.

### 5. Keeping Your Branch Updated

`main` will continuously receive changes from other sub-teams.

Each sub-team should pull the latest `main` regularly when applicable, especially before integration or opening a PR.

Recommended:

```bash
git checkout main
git pull origin main

git checkout feature/<main-task>
git merge main
```

Resolve any conflicts locally, test again, and push the updated branch.

For this project, prefer simple Git operations over complicated branching strategies.

### 6. Commits

Keep commits small and descriptive.

**Format:**

```
<type>: <description>
```

Examples:

- `feat: add semantic chunking`
- `feat: implement vector indexing`
- `fix: preserve document metadata`
- `test: add retrieval tests`
- `docs: update API contract`
- `refactor: simplify embedding pipeline`

Avoid vague commits such as:

- `update`
- `changes`
- `stuff`
- `final`

Commits should clearly describe what was changed.

### 7. API Contracts

If a component exposes an API or communicates with another component through an API, its contract must be documented in:

```
/docs/api-contract.md
```

This file is the single source of truth for the project's API interfaces.

The responsible sub-team should document, where applicable:

- Endpoint
- HTTP method
- Request schema
- Response schema
- Required fields
- Data types
- Example request
- Example response
- Error responses
- Important assumptions

For example:

```
POST /retrieve

Request:
{
  "query": "..."
}

Response:
{
  "results": [...]
}
```

Not every main task needs an API contract. Only components that expose or consume an API need to be documented there.

If an API contract changes, update:

- `/docs/api-contract.md`

and make sure affected sub-teams are aware of the change.

### 8. Mock Data

The repository contains:

```
/mock-data
```

This folder contains example:

- Requests
- Responses
- Outputs
- Inputs
- Intermediate data

that other sub-teams can use while developing their components.

The purpose is to allow teams to work in parallel without waiting for another component to be completed.

For example:

```
Component A
     ↓
Component B
```

If Component A is not ready yet, Component B can use the corresponding example output from:

```
/mock-data
```

Once Component A is ready, Component B can replace the mock with the real implementation.

**Important**

If you introduce a new data shape that another sub-team will depend on:

- Add or update the corresponding mock data.
- Document the interface in `/docs/api-contract.md` if it is an API.
- Inform the affected sub-team.
- Keep the mock data consistent with the agreed interface.

Mock data should represent the actual expected structure, not an arbitrary example.

### 9. Integration

Components should communicate through their defined interfaces whenever possible.

For components that expose APIs, use the actual API during integration.

For example:

```
Component A
     ↓
   API
     ↓
Component B
```

During development, mocks can replace unavailable components.

During integration, replace the mocks with the real implementation and verify that:

- Inputs match the expected structure
- Outputs match the expected structure
- Errors are handled
- Required metadata is preserved
- The components work together as expected

For API-based components such as FastAPI services, test the actual HTTP interface where possible rather than only testing internal functions.

For non-API components, test the actual agreed input/output interface instead.

### 10. Documentation and Outputs

Each sub-team is responsible for keeping relevant documentation and outputs up to date.

Use:

```
/docs
```

for shared project documentation, including:

- `/docs/api-contract.md`

Service/workstream-specific documentation should be added where appropriate.

Generated examples, test outputs, and shared outputs should be placed in the appropriate project folders rather than committed randomly throughout the repository.

### 11. Dependencies

If your work depends on another component or Issue, document it clearly.

Example:

```
Depends on: #12
```

Dependencies should not unnecessarily block development.

If the expected interface has already been agreed upon, use the corresponding mock data and continue development independently.

---

## Golden Rule

Each sub-team owns a main-task/workstream branch, Issues track the work implemented within that branch, interfaces are documented and mocked when needed, and only fully tested and integratable work reaches main through a Pull Request.
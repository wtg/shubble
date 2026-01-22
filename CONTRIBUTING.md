# Contributing to Shubble

Thank you for your interest in contributing to Shubble! This project is developed under Rensselaer Polytechnic Institute's Rensselaer Center for Open Source (RCOS), and we welcome contributions from everyone.

## Ways to Contribute

- **Code** - Bug fixes, new features, performance improvements
- **Documentation** - Improve guides, add examples, fix typos
- **Design** - UI/UX improvements, accessibility enhancements
- **Testing** - Report bugs, write tests, improve test coverage
- **Ideas** - Feature suggestions, architecture discussions

## Getting Started

### 1. Installation

See **[docs/INSTALLATION.md](docs/INSTALLATION.md)** for complete setup instructions, including:

- Docker-based setup (recommended for quick start)
- Host-based setup (recommended for active development)
- Mixed setups (Docker for some services, host for others)
- Environment variable configuration

### 2. Project Structure

See **[README.md](README.md)** for an overview of the project structure. Each major directory has its own README with detailed documentation:

- [backend/README.md](backend/README.md) - FastAPI backend, API endpoints, database schema
- [frontend/README.md](frontend/README.md) - React frontend, components, configuration
- [ml/README.md](ml/README.md) - Machine learning pipelines, model training
- [test/README.md](test/README.md) - Test environment

### 3. Testing

See **[docs/TESTING.md](docs/TESTING.md)** for instructions on:

- Running the test environment with Docker Compose
- Using the mock Samsara API for development
- Testing without real API credentials

## Development Workflow

### Opening Issues

Found a bug? Have a feature idea? **Please open an issue!**

- **Bug reports** - Include steps to reproduce, expected vs actual behavior
- **Feature requests** - Describe the use case and proposed solution
- **Questions** - Ask about architecture, implementation details, or getting started

Browse existing issues: [github.com/wtg/shubble/issues](https://github.com/wtg/shubble/issues)

### Submitting Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Test your changes** locally (see [docs/TESTING.md](docs/TESTING.md))
4. **Open a pull request** with a clear description of the changes

#### PR Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new functionality
- Follow existing code style and conventions

### Code Review

All PRs require review before merging. Reviewers will check for:

- Code quality and readability
- Test coverage
- Documentation updates
- Adherence to project conventions

## Database Migrations

When modifying the database schema:

1. Update models in `backend/models.py`
2. Generate a migration:
   ```bash
   uv run alembic -c backend/alembic.ini revision --autogenerate -m "Description"
   ```
3. Apply the migration:
   ```bash
   uv run alembic -c backend/alembic.ini upgrade head
   ```
4. Commit the migration file in `alembic/versions/`

## Staging Environment

For testing changes that require external services (MapKit, Samsara API):

- **Staging URL**: [https://staging-web-shuttles.rpi.edu](https://staging-web-shuttles.rpi.edu)
- Deploy via GitHub Actions: Actions > Deploy to Staging > Run workflow
- Notify other developers on Discord before using staging

## Communication

- **Discord** - Shubble Developers server (for real-time discussion)
- **GitHub Issues** - Bug reports, feature requests, questions
- **GitHub Discussions** - Architecture discussions, RFCs

## Questions?

Don't hesitate to ask! Open an issue or reach out on Discord. We're happy to help new contributors get started.

---

**Live site**: [https://shuttles.rpi.edu](https://shuttles.rpi.edu)

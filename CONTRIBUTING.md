# Contributing to Shubble

Thank you for your interest in contributing to Shubble! This project is part of Rensselaer Polytechnic Institute's Rensselaer Center for Open Source (RCOS), and we welcome all contributions.

## Getting Started

### 1. Set Up Your Development Environment

Follow the setup instructions in [docs/INSTALLATION.md](docs/INSTALLATION.md) to get Shubble running locally. We recommend the native setup for active development.

**Quick Start:**
```bash
# Clone the repository
git clone git@github.com:wtg/shubble.git
cd shubble

# Follow native or Docker setup in INSTALLATION.md
```

### 2. Find Something to Work On

**Good First Steps:**
- Browse open issues on GitHub labeled `good first issue`
- Check for documentation improvements
- Fix bugs or add tests for existing features
- Improve error messages or user experience

**Ideas for Contributions:**
- **Frontend**: UI improvements, new visualizations, mobile experience
- **Backend**: API optimizations, new endpoints, data analysis features
- **Algorithm**: Schedule matching improvements, route prediction
- **Testing**: Add test coverage, improve test infrastructure
- **Documentation**: Clarify setup steps, add code comments, create guides
- **DevOps**: Docker improvements, CI/CD enhancements

### 3. Development Workflow

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following existing code patterns:
   - Use native setup for faster iteration
   - Refer to [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development commands
   - Follow the existing code style and conventions

3. **Test your changes**:
   ```bash
   # Frontend tests
   npm test

   # Backend tests
   pytest

   # Linting
   npm run lint
   ```

   See [testing/README.md](testing/README.md) for comprehensive testing guide.

4. **Commit your changes** with clear commit messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description of what you did"
   ```

5. **Push to your fork** and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub:
   - Describe what your PR does
   - Reference any related issues
   - Include screenshots for UI changes
   - Explain how you tested the changes

## Development Guidelines

### Code Quality

**Before committing:**
- Run tests and ensure they pass
- Run linter and fix any issues
- Test manually in your browser/with API calls
- Ensure your changes don't break existing functionality

**Code Standards:**
- Follow existing patterns in the codebase
- Write clear, descriptive variable and function names
- Add comments for complex logic
- Keep functions focused and single-purpose
- Update documentation when adding features

### Testing

- Add tests for new features
- Update tests when modifying existing code
- Ensure all tests pass before submitting PR
- See [testing/README.md](testing/README.md) for testing guide

### Documentation

Update documentation when:
- Adding new features or API endpoints
- Changing setup/configuration requirements
- Modifying database schema
- Adding environment variables
- Changing deployment process

## Pull Request Process

1. **Create PR** with clear title and description
2. **Link related issues** using "Fixes #123" syntax
3. **Wait for review** from maintainers
4. **Address feedback** by pushing new commits
5. **Merge** once approved (maintainer will merge)

**PR Checklist:**
- [ ] Tests pass (`npm test && pytest`)
- [ ] Linting passes (`npm run lint`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear
- [ ] No merge conflicts with main branch

## Getting Help

**Need assistance?**
- Check documentation in [docs/](docs/) directory
- Review [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for common commands
- Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Open a GitHub issue with questions
- Contact: [Joel McCandless](mailto:mail@joelmccandless.com)

**Resources:**
- [INSTALLATION.md](docs/INSTALLATION.md) - Setup instructions
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development commands and workflows
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and structure
- [testing/README.md](testing/README.md) - Testing guide (unit tests, mock API)
- [PORTS.md](PORTS.md) - Port reference for all services

## Code of Conduct

Be respectful and inclusive:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Questions?

Don't hesitate to ask questions! We're here to help new contributors get started. Open an issue or reach out directly.

---

**Thank you for contributing to Shubble! 🚌**

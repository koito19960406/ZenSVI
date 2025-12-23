# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `zensvi` for local development.

1. Download a copy of `zensvi` locally.

2. Create a virtual environment using conda:

    ```console
    $ conda create -n zensvi-env python=3.9
    $ conda activate zensvi-env
    ```

3. Install Poetry using pip:

    ```console
    $ pip install poetry
    ```

4. Install `zensvi` using `poetry`:

    ```console
    $ poetry install
    ```

5. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

4. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

5. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests that cover the new functionality or bug fix.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for all currently supported Python versions. Check https://github.com/koito19960406/ZenSVI/actions and make sure that the tests pass for all supported Python versions.
4. If your change affects the public API, update the docstrings and ensure they render correctly in the documentation.

## Code Style

This project uses `black` for code formatting, `isort` for import sorting, and `flake8` for linting. Please ensure your code adheres to these standards before submitting a pull request.

To format your code, run:
```
black src
isort src
```

To check linting:
```
flake8 src
```

## Commit Messages

Please follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Code of Conduct

Please note that the `zensvi` project is released with a [Code of Conduct](CONDUCT.md). By contributing to this project you agree to abide by its terms.

## Docstring Style

We use Google-style docstrings for this project. Please ensure all functions, classes, and modules have proper docstrings. Here's an example:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.

    Longer description of the function if necessary.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ExceptionType: When and why this exception is raised.
    """
    # Function body
```

To help maintain consistent docstring formatting, please use the following tools:

1. Use `docformatter` to format docstrings:
   ```
   docformatter --in-place --wrap-summaries 88 --wrap-descriptions 88 -r src
   ```

2. Use `docconvert` to convert docstrings to Google style:
   ```
   docconvert -i guess -o google --in-place -v src
   ```

3. Use `pydocstyle` to check docstring formatting:
   ```
   pydocstyle src
   ```

Make sure to run these tools before submitting your code. This will help ensure consistent and well-formatted docstrings throughout the project.

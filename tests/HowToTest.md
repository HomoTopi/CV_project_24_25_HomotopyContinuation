# How to develop test for this project
This is a short guide on how to develop tests for this project. The project uses the pytest framework for testing. The tests are located in this directory: `tests/`.

## How to write a test
Writing a test is simple and can be done in a few steps. Here is an example of how to write a test for the `add` function in the `calculator` module.

1. Create a new file in the `tests/ModuleName/` directory. The file should have the name `test_<module_name>.py`. In this case, the file should be named `test_calculator.py`.

2. Import the `calculator` module and the `pytest` module.

```python
import calculator
import pytest
```

3. Write the test function. The test function should start with the word `test_` and should include both the target method and the expected behavior.

```python
def test_add_theAdditionShouldWork():
    assert calculator.add(1, 2) == 3
    assert calculator.add(0, 0) == 0
    assert calculator.add(-1, 1) == 0
```

## How to run the tests
To run the tests, you need to execute the following command in the root directory of the project:

```bash
pytest
```

import pytest
import requests


def ollama_up():
    try:
        return requests.get("http://localhost:11434/api/version", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.fixture
def needs_ollama():
    if not ollama_up():
        pytest.skip("Ollama is not running on localhost:11434")

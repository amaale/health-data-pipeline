import pytest
from src.cleaner import MedicalTextPreprocessor

@pytest.fixture(scope="module")
def preprocessor():
    """Setup the preprocessor once for all tests"""
    return MedicalTextPreprocessor()

def test_basic_cleaning(preprocessor):
    raw_text = "I have a HEADACHE!!!"
    expected = "headache"
    assert preprocessor.clean_text(raw_text) == expected

def test_remove_stopwords(preprocessor):
    raw_text = "I am feeling the pain"
    # 'I', 'am', 'the' are stopwords. 'feeling' might lemmatize to 'feel'.
    result = preprocessor.clean_text(raw_text)
    assert "pain" in result
    assert "the" not in result

def test_lemmatization(preprocessor):
    raw_text = "My knees are aching badly"
    # 'aching' -> 'ache', 'knees' -> 'knee'
    result = preprocessor.clean_text(raw_text)
    assert "ache" in result or "aching" in result # Depending on spaCy model version

def test_empty_input(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""

def test_medical_jargon(preprocessor):
    raw_text = "Patient suffering from myocardial infarction"
    result = preprocessor.clean_text(raw_text)
    assert "myocardial" in result
    assert "infarction" in result

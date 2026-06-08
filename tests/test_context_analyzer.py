"""Tests for the context analyzer."""

import pytest
from token_tamer.context_analyzer import ContextAnalyzer
from token_tamer.skeletonizer import Skeletonizer


@pytest.fixture
def analyzer():
    return ContextAnalyzer(Skeletonizer())


class TestActiveFileDetection:
    def test_explicit_filename_in_prompt(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": (
                    "Fix the bug in payment.py where tax calculation is wrong.\n\n"
                    "```python\n# File: payment.py\ndef calc(): pass\n```\n\n"
                    "```python\n# File: database.py\ndef get_conn(): pass\n```"
                ),
            }
        ]
        active = analyzer.extract_active_files(messages)
        assert "payment.py" in active
        assert "database.py" not in active

    def test_no_filename_means_intact(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": "```python\ndef foo():\n    return 42\n```",
            }
        ]
        compressed, result = analyzer.analyze_and_compress(messages)
        assert result.skeletonized_blocks == 0
        for block in result.code_blocks:
            assert block.is_active is True

    def test_path_normalization(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": "Fix src/utils/payment.py",
            }
        ]
        active = analyzer.extract_active_files(messages)
        assert "payment.py" in active

    def test_anthropic_system_prompt_active_files(self, analyzer: ContextAnalyzer):
        messages = [{"role": "user", "content": "Do something"}]
        all_messages = [
            {"role": "system", "content": "Focus on payment.py"},
            {"role": "user", "content": "Do something"},
        ]
        active = analyzer.extract_active_files(all_messages)
        assert "payment.py" in active


class TestCompression:
    def test_background_file_gets_skeletonized(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": (
                    "Fix the math in payment.py\n\n"
                    "```python\n# File: payment.py\ndef calc():\n    return 1+1\n```\n\n"
                    "```python\n# File: database.py\ndef connect():\n    x = 1\n    y = 2\n    return x+y\n```"
                ),
            }
        ]
        compressed, result = analyzer.analyze_and_compress(messages)
        assert result.total_blocks == 2
        assert result.skeletonized_blocks == 1

        db_block = [b for b in result.code_blocks if b.filename == "database.py"][0]
        assert db_block.is_active is False
        assert db_block.skeleton_result is not None
        assert db_block.skeleton_result.was_compressed is True

        pay_block = [b for b in result.code_blocks if b.filename == "payment.py"][0]
        assert pay_block.is_active is True
        assert pay_block.skeleton_result is None

    def test_multipart_content_handling(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fix payment.py"},
                    {"type": "text", "text": "```python\n# File: utils.py\ndef helper():\n    return 42\n```"},
                ],
            }
        ]
        compressed, result = analyzer.analyze_and_compress(messages)
        assert result.total_blocks == 1
        assert result.skeletonized_blocks == 1


class TestMultiLanguage:
    def test_javascript_background_skeletonized(self, analyzer: ContextAnalyzer):
        messages = [
            {
                "role": "user",
                "content": (
                    "Fix payment.js\n\n"
                    "```javascript\n// File: payment.js\nfunction calc() { return 1; }\n```\n\n"
                    "```javascript\n// File: utils.js\nfunction helper() {\n  console.log(1);\n  console.log(2);\n  return 42;\n}\n```"
                ),
            }
        ]
        compressed, result = analyzer.analyze_and_compress(messages)
        assert result.skeletonized_blocks >= 1

        utils_block = [b for b in result.code_blocks if b.filename == "utils.js"][0]
        assert utils_block.is_active is False
        assert utils_block.skeleton_result is not None

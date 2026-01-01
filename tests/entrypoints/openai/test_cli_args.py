# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging

import pytest

from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.logging_utils import AccessLogPathFilter
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ...utils import VLLM_PATH

LORA_MODULE = {
    "name": "module2",
    "path": "/path/to/module2",
    "base_model_name": "llama",
}
CHATML_JINJA_PATH = VLLM_PATH / "examples/template_chatml.jinja"
assert CHATML_JINJA_PATH.exists()


@pytest.fixture
def serve_parser():
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    return make_arg_parser(parser)


### Test config parsing
def test_config_arg_parsing(serve_parser, cli_config_file):
    args = serve_parser.parse_args([])
    assert args.port == 8000
    args = serve_parser.parse_args(["--config", cli_config_file])
    assert args.port == 12312
    args = serve_parser.parse_args(
        [
            "--config",
            cli_config_file,
            "--port",
            "9000",
        ]
    )
    assert args.port == 9000
    args = serve_parser.parse_args(
        [
            "--port",
            "9000",
            "--config",
            cli_config_file,
        ]
    )
    assert args.port == 9000


### Tests for LoRA module parsing
def test_valid_key_value_format(serve_parser):
    # Test old format: name=path
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
        ]
    )
    expected = [LoRAModulePath(name="module1", path="/path/to/module1")]
    assert args.lora_modules == expected


def test_valid_json_format(serve_parser):
    # Test valid JSON format input
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module2", path="/path/to/module2", base_model_name="llama")
    ]
    assert args.lora_modules == expected


def test_invalid_json_format(serve_parser):
    # Test invalid JSON format input, missing closing brace
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            ["--lora-modules", '{"name": "module3", "path": "/path/to/module3"']
        )


def test_invalid_type_error(serve_parser):
    # Test type error when values are not JSON or key=value
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                "invalid_format",  # This is not JSON or key=value format
            ]
        )


def test_invalid_json_field(serve_parser):
    # Test valid JSON format but missing required fields
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                '{"name": "module4"}',  # Missing required 'path' field
            ]
        )


def test_empty_values(serve_parser):
    # Test when no LoRA modules are provided
    args = serve_parser.parse_args(["--lora-modules", ""])
    assert args.lora_modules == []


def test_multiple_valid_inputs(serve_parser):
    # Test multiple valid inputs (both old and JSON format)
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module1", path="/path/to/module1"),
        LoRAModulePath(
            name="module2", path="/path/to/module2", base_model_name="llama"
        ),
    ]
    assert args.lora_modules == expected


### Tests for serve argument validation that run prior to loading
def test_enable_auto_choice_passes_without_tool_call_parser(serve_parser):
    """Ensure validation fails if tool choice is enabled with no call parser"""
    # If we enable-auto-tool-choice, explode with no tool-call-parser
    args = serve_parser.parse_args(args=["--enable-auto-tool-choice"])
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_enable_auto_choice_passes_with_tool_call_parser(serve_parser):
    """Ensure validation passes with tool choice enabled with a call parser"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "mistral",
        ]
    )
    validate_parsed_serve_args(args)


def test_enable_auto_choice_fails_with_enable_reasoning(serve_parser):
    """Ensure validation fails if reasoning is enabled with auto tool choice"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_passes_with_reasoning_parser(serve_parser):
    """Ensure validation passes if reasoning is enabled
    with a reasoning parser"""
    args = serve_parser.parse_args(
        args=[
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_happy_paths(serve_parser):
    """Ensure validation passes if the chat template exists"""
    args = serve_parser.parse_args(
        args=["--chat-template", CHATML_JINJA_PATH.absolute().as_posix()]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_sad_paths(serve_parser):
    """Ensure validation fails if the chat template doesn't exist"""
    args = serve_parser.parse_args(args=["--chat-template", "does/not/exist"])
    with pytest.raises(ValueError):
        validate_parsed_serve_args(args)


@pytest.mark.parametrize(
    "cli_args, expected_middleware",
    [
        (
            ["--middleware", "middleware1", "--middleware", "middleware2"],
            ["middleware1", "middleware2"],
        ),
        ([], []),
    ],
)
def test_middleware(serve_parser, cli_args, expected_middleware):
    """Ensure multiple middleware args are parsed properly"""
    args = serve_parser.parse_args(args=cli_args)
    assert args.middleware == expected_middleware


def test_default_chat_template_kwargs_parsing(serve_parser):
    """Ensure default_chat_template_kwargs JSON is parsed correctly"""
    args = serve_parser.parse_args(
        args=["--default-chat-template-kwargs", '{"enable_thinking": false}']
    )
    assert args.default_chat_template_kwargs == {"enable_thinking": False}


def test_default_chat_template_kwargs_complex(serve_parser):
    """Ensure complex default_chat_template_kwargs JSON is parsed correctly"""
    kwargs_json = '{"enable_thinking": false, "custom_param": "value", "num": 42}'
    args = serve_parser.parse_args(args=["--default-chat-template-kwargs", kwargs_json])
    assert args.default_chat_template_kwargs == {
        "enable_thinking": False,
        "custom_param": "value",
        "num": 42,
    }


def test_default_chat_template_kwargs_default_none(serve_parser):
    """Ensure default_chat_template_kwargs defaults to None"""
    args = serve_parser.parse_args(args=[])
    assert args.default_chat_template_kwargs is None


def test_default_chat_template_kwargs_invalid_json(serve_parser):
    """Ensure invalid JSON raises an error"""
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            args=["--default-chat-template-kwargs", "not valid json"]
        )


### Tests for uvicorn_access_log_path_filter argument
def test_access_log_path_filter_single_path(serve_parser):
    """Test parsing a single path for access log filtering"""
    args = serve_parser.parse_args(
        args=["--uvicorn-access-log-path-filter", "/metrics"]
    )
    assert args.uvicorn_access_log_path_filter == ["/metrics"]


def test_access_log_path_filter_multiple_paths(serve_parser):
    """Test parsing multiple comma-separated paths for access log filtering"""
    args = serve_parser.parse_args(
        args=["--uvicorn-access-log-path-filter", "/metrics,/health,/ready"]
    )
    assert args.uvicorn_access_log_path_filter == ["/metrics", "/health", "/ready"]


def test_access_log_path_filter_with_spaces(serve_parser):
    """Test that spaces around commas are handled correctly"""
    args = serve_parser.parse_args(
        args=["--uvicorn-access-log-path-filter", "/metrics , /health , /ready"]
    )
    assert args.uvicorn_access_log_path_filter == ["/metrics", "/health", "/ready"]


def test_access_log_path_filter_default_none(serve_parser):
    """Test that default value is None when not specified"""
    args = serve_parser.parse_args(args=[])
    assert args.uvicorn_access_log_path_filter is None


def test_access_log_path_filter_empty_string(serve_parser):
    """Test that empty string results in empty list"""
    args = serve_parser.parse_args(
        args=["--uvicorn-access-log-path-filter", ""]
    )
    assert args.uvicorn_access_log_path_filter == []


### Tests for AccessLogPathFilter class
class TestAccessLogPathFilter:
    """Tests for the AccessLogPathFilter logging filter"""

    def _create_log_record(self, message: str) -> logging.LogRecord:
        """Helper to create a mock log record with a specific message"""
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        return record

    def test_filter_allows_when_no_excluded_paths(self):
        """Test that filter allows all logs when no paths are excluded"""
        filter_ = AccessLogPathFilter()
        record = self._create_log_record(
            '127.0.0.1 - "GET /metrics HTTP/1.1" 200'
        )
        assert filter_.filter(record) is True

    def test_filter_blocks_metrics_path(self):
        """Test that filter blocks /metrics endpoint logs"""
        filter_ = AccessLogPathFilter(["/metrics"])
        record = self._create_log_record(
            '127.0.0.1 - "GET /metrics HTTP/1.1" 200'
        )
        assert filter_.filter(record) is False

    def test_filter_blocks_health_path(self):
        """Test that filter blocks /health endpoint logs"""
        filter_ = AccessLogPathFilter(["/health"])
        record = self._create_log_record(
            '127.0.0.1 - "GET /health HTTP/1.1" 200'
        )
        assert filter_.filter(record) is False

    def test_filter_allows_non_excluded_paths(self):
        """Test that filter allows logs for non-excluded paths"""
        filter_ = AccessLogPathFilter(["/metrics", "/health"])
        record = self._create_log_record(
            '127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200'
        )
        assert filter_.filter(record) is True

    def test_filter_blocks_multiple_excluded_paths(self):
        """Test that filter blocks all excluded paths"""
        filter_ = AccessLogPathFilter(["/metrics", "/health", "/ready"])

        # Test each excluded path
        for path in ["/metrics", "/health", "/ready"]:
            record = self._create_log_record(
                f'127.0.0.1 - "GET {path} HTTP/1.1" 200'
            )
            assert filter_.filter(record) is False

    def test_filter_handles_path_with_query_string(self):
        """Test that filter blocks paths with query strings"""
        filter_ = AccessLogPathFilter(["/metrics"])
        record = self._create_log_record(
            '127.0.0.1 - "GET /metrics?format=json HTTP/1.1" 200'
        )
        assert filter_.filter(record) is False

    def test_filter_with_empty_excluded_paths(self):
        """Test that filter allows all logs when excluded_paths is empty"""
        filter_ = AccessLogPathFilter([])
        record = self._create_log_record(
            '127.0.0.1 - "GET /metrics HTTP/1.1" 200'
        )
        assert filter_.filter(record) is True

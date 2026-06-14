import json
import unittest
from unittest.mock import MagicMock, patch

from defender.utils import (
    _compare_guess_to_ground_truth,
    is_guess_correct,
    parse_llm_json,
    parse_validated_llm_json,
    validate_defender_response,
    validate_utility_response,
)
from defender.types import (
    AdversarialIteration,
    AdversarialResult,
    AttackerOutput,
    DefenderInput,
    DefenderOutput,
    StrategyRecord,
    UtilityInput,
    UtilityOutput,
)
from defender.attacker import Attacker, AttackerError
from defender.defender import Defender, DefenderError, _normalize_clue_map
from defender.llm_client import GeminiClient
from defender.orchestrator import (
    run_adversarial_loop,
    _build_attacker_feedback,
    _build_iteration_feedback,
    _verified_successful_attributes,
)
from defender.prompts import (
    build_clue_enumeration_prompt,
    build_ground_truth_prompt,
    build_rewrite_prompt,
    build_utility_prompt,
)
from defender.scanner import PIIMatch, _deduplicate_spans, _is_valid_phone, _luhn_check, scan_text
from defender.strategies import RewriteStrategy, VALID_STRATEGY_NAMES
from defender.utility import UtilityError, UtilityJudge


class TestUtils(unittest.TestCase):
    def test_compare_guess_to_ground_truth(self) -> None:
        self.assertTrue(_compare_guess_to_ground_truth("46", "46"))
        self.assertTrue(_compare_guess_to_ground_truth("46 years old", "46"))
        self.assertTrue(_compare_guess_to_ground_truth("46", "46 years old"))
        self.assertTrue(_compare_guess_to_ground_truth("Cardiologist", "cardiologist"))
        self.assertTrue(_compare_guess_to_ground_truth("mid-40s", "46", "Age"))
        self.assertTrue(
            _compare_guess_to_ground_truth("born around 1960", "1963", "Birth Year")
        )

        self.assertFalse(_compare_guess_to_ground_truth("UNKNOWN", "46"))
        self.assertFalse(_compare_guess_to_ground_truth("46", "UNKNOWN"))
        self.assertFalse(_compare_guess_to_ground_truth("", "46"))
        self.assertFalse(_compare_guess_to_ground_truth("46", ""))
        self.assertFalse(_compare_guess_to_ground_truth("mid-40s", "46", "Profession"))
        self.assertFalse(_compare_guess_to_ground_truth("1960", "1963", "Location"))
        self.assertFalse(_compare_guess_to_ground_truth("District 4", "District 9", "Location"))
        self.assertFalse(_compare_guess_to_ground_truth("London", "Paris"))
        self.assertFalse(_compare_guess_to_ground_truth("1990", "2000"))

    def test_compare_guess_year_boundary(self) -> None:
        """Birth year tolerance is 5 — exactly 5 apart should match, 6 should not."""
        self.assertTrue(_compare_guess_to_ground_truth("1958", "1963", "Birth Year"))
        self.assertFalse(_compare_guess_to_ground_truth("1956", "1963", "Birth Year"))

    def test_compare_guess_age_boundary(self) -> None:
        """Age tolerance is 6 — exactly 6 apart should match, 7 should not."""
        self.assertTrue(_compare_guess_to_ground_truth("40", "46", "Age"))
        self.assertFalse(_compare_guess_to_ground_truth("39", "46", "Age"))

    def test_is_guess_correct_none_truth(self) -> None:
        self.assertFalse(is_guess_correct("46", None, 0.9, "Age"))

    def test_validate_utility_response(self) -> None:
        self.assertEqual(validate_utility_response({"score": 0.8, "rationale": "Good"}), [])
        self.assertIn(
            "score",
            "".join(validate_utility_response({"score": -0.1, "rationale": "Good"})),
        )
        self.assertIn(
            "score",
            "".join(validate_utility_response({"score": "0.8", "rationale": "Good"})),
        )
        self.assertIn("rationale", "".join(validate_utility_response({"score": 0.8})))
        self.assertIn(
            "rationale",
            "".join(validate_utility_response({"score": 0.8, "rationale": "   "})),
        )

    def test_validate_utility_response_score_above_1(self) -> None:
        errors = validate_utility_response({"score": 1.1, "rationale": "Good"})
        self.assertTrue(any("score" in e for e in errors))

    def test_validate_defender_response_rejects_empty_rewrite(self) -> None:
        errors = validate_defender_response(
            {
                "rewritten_text": "   ",
                "strategies_used": [
                    {
                        "attribute": "Age",
                        "strategy": "abstraction",
                        "reasoning": "Removed age clues.",
                    }
                ],
                "confidence": 0.8,
            },
            ["Age"],
        )

        self.assertIn("non-empty", " ".join(errors))

    def test_validate_defender_response_valid(self) -> None:
        errors = validate_defender_response(
            {
                "rewritten_text": "Some text",
                "strategies_used": [
                    {
                        "attribute": "Age",
                        "strategy": "abstraction",
                        "reasoning": "Removed age.",
                    }
                ],
                "confidence": 0.8,
            },
            ["Age"],
        )
        self.assertEqual(errors, [])

    def test_validate_defender_response_invalid_strategy_name(self) -> None:
        errors = validate_defender_response(
            {
                "rewritten_text": "Text",
                "strategies_used": [
                    {
                        "attribute": "Age",
                        "strategy": "deletion",
                        "reasoning": "Bad strategy.",
                    }
                ],
                "confidence": 0.8,
            },
            ["Age"],
        )
        self.assertTrue(any("invalid strategy" in e for e in errors))

    def test_validate_defender_response_missing_target_coverage(self) -> None:
        errors = validate_defender_response(
            {
                "rewritten_text": "Text",
                "strategies_used": [
                    {
                        "attribute": "Age",
                        "strategy": "abstraction",
                        "reasoning": "OK.",
                    }
                ],
                "confidence": 0.8,
            },
            ["Age", "Location"],
        )
        self.assertTrue(any("Location" in e for e in errors))

    def test_validate_defender_response_normalizes_strategy_case(self) -> None:
        data: dict[str, object] = {
            "rewritten_text": "Text",
            "strategies_used": [
                {
                    "attribute": "Age",
                    "strategy": "ABSTRACTION",
                    "reasoning": "OK.",
                }
            ],
            "confidence": 0.8,
        }
        errors = validate_defender_response(data, ["Age"])
        self.assertEqual(errors, [])
        strategies = data["strategies_used"]
        assert isinstance(strategies, list)
        self.assertEqual(strategies[0]["strategy"], "abstraction")

    def test_parse_llm_json_ignores_malformed_brace_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "No JSON object found"):
            parse_llm_json("prefix {not valid json} suffix")

    def test_parse_llm_json_plain(self) -> None:
        result = parse_llm_json('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_parse_llm_json_markdown_fences(self) -> None:
        result = parse_llm_json('```json\n{"key": "value"}\n```')
        self.assertEqual(result, {"key": "value"})

    def test_parse_llm_json_surrounding_text(self) -> None:
        result = parse_llm_json('Here is the result: {"key": "value"} hope that helps!')
        self.assertEqual(result, {"key": "value"})

    def test_parse_llm_json_empty_string(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_llm_json("")

    def test_parse_validated_llm_json_success_first_try(self) -> None:
        result = parse_validated_llm_json(
            initial_text='{"score": 0.8, "rationale": "Good"}',
            retry=lambda: "",
            validate=validate_utility_response,
            error_type=ValueError,
            parse_error_message="Parse failed: {error}",
            validation_error_message="Validation failed: {errors}",
        )
        self.assertEqual(result["score"], 0.8)

    def test_parse_validated_llm_json_retries_on_parse_failure(self) -> None:
        result = parse_validated_llm_json(
            initial_text="not json",
            retry=lambda: '{"score": 0.8, "rationale": "Good"}',
            validate=validate_utility_response,
            error_type=ValueError,
            parse_error_message="Parse failed: {error}",
            validation_error_message="Validation failed: {errors}",
        )
        self.assertEqual(result["score"], 0.8)

    def test_parse_validated_llm_json_retries_on_validation_failure(self) -> None:
        result = parse_validated_llm_json(
            initial_text='{"score": 1.5, "rationale": "Bad"}',
            retry=lambda: '{"score": 0.8, "rationale": "Good"}',
            validate=validate_utility_response,
            error_type=ValueError,
            parse_error_message="Parse failed: {error}",
            validation_error_message="Validation failed: {errors}",
        )
        self.assertEqual(result["score"], 0.8)

    def test_parse_validated_llm_json_raises_after_double_failure(self) -> None:
        with self.assertRaises(ValueError):
            parse_validated_llm_json(
                initial_text="not json",
                retry=lambda: "still not json",
                validate=validate_utility_response,
                error_type=ValueError,
                parse_error_message="Parse failed: {error}",
                validation_error_message="Validation failed: {errors}",
            )

    def test_is_guess_correct_consolidated(self) -> None:
        self.assertTrue(is_guess_correct("mid-40s", "46", 0.7, "Age"))
        self.assertFalse(is_guess_correct("mid-40s", "46", 0.69, "Age"))
        self.assertFalse(is_guess_correct("UNKNOWN", "46", 0.9, "Age"))
        self.assertFalse(is_guess_correct("District 4", "District 9", 0.9, "Location"))

    def test_is_guess_correct_custom_threshold(self) -> None:
        self.assertTrue(is_guess_correct("46", "46", 0.8, "Age", threshold=0.8))
        self.assertFalse(is_guess_correct("46", "46", 0.79, "Age", threshold=0.8))


class TestScanner(unittest.TestCase):
    def test_scan_text_masks_regex_pii(self) -> None:
        result = scan_text("Email sarah@example.com and card 4111 1111 1111 1111.")

        self.assertIn("<EMAIL>", result.masked_text)
        self.assertIn("<CREDIT_CARD>", result.masked_text)
        self.assertNotIn("sarah@example.com", result.masked_text)

    def test_scan_text_empty_input(self) -> None:
        result = scan_text("")
        self.assertEqual(result.masked_text, "")
        self.assertEqual(result.pii_found, [])

    def test_scan_text_no_pii(self) -> None:
        text = "The sky is blue and the grass is green."
        result = scan_text(text)
        self.assertEqual(result.masked_text, text)

    def test_scan_text_date_detected_but_not_masked(self) -> None:
        result = scan_text("Event on 2024-01-15 was great.")
        date_matches = [m for m in result.pii_found if m.pii_type == "DATE"]
        self.assertTrue(len(date_matches) > 0)
        # Dates should NOT be masked
        self.assertIn("2024-01-15", result.masked_text)

    def test_luhn_check_valid(self) -> None:
        self.assertTrue(_luhn_check("4111111111111111"))

    def test_luhn_check_invalid(self) -> None:
        self.assertFalse(_luhn_check("4111111111111112"))

    def test_luhn_check_non_numeric(self) -> None:
        self.assertFalse(_luhn_check("abc"))

    def test_deduplicate_spans_prioritizes_maskable_overlaps(self) -> None:
        non_maskable = PIIMatch(
            pii_type="ORGANIZATION",
            value="Contact sarah@example.com",
            start=0,
            end=25,
            replacement="<ORGANIZATION>",
            mask=False,
        )
        maskable = PIIMatch(
            pii_type="EMAIL",
            value="sarah@example.com",
            start=8,
            end=25,
            replacement="<EMAIL>",
            mask=True,
        )

        result = _deduplicate_spans([non_maskable, maskable])

        self.assertEqual(result, [maskable])

    def test_deduplicate_spans_empty(self) -> None:
        self.assertEqual(_deduplicate_spans([]), [])

    def test_is_valid_phone_local_number_fallback(self) -> None:
        self.assertTrue(_is_valid_phone("(202) 555-1234"))


class TestTypes(unittest.TestCase):
    def test_utility_input_from_dict_defaults(self) -> None:
        result = UtilityInput.from_dict({})

        self.assertEqual(result.original_text, "")
        self.assertEqual(result.rewritten_text, "")
        self.assertIsNone(result.target_attributes)

    def test_utility_input_from_dict_target_attributes(self) -> None:
        result = UtilityInput.from_dict(
            {
                "original_text": "Original",
                "rewritten_text": "Rewritten",
                "target_attributes": ["Age", 46],
            }
        )

        self.assertEqual(result.to_dict()["target_attributes"], ["Age", "46"])

    def test_defender_input_roundtrip(self) -> None:
        original = DefenderInput(
            text="Some text",
            target_attributes=["Age", "Location"],
            iteration=2,
            attacker_feedback="feedback",
            ground_truth={"Age": "46"},
        )
        restored = DefenderInput.from_dict(original.to_dict())
        self.assertEqual(restored.text, original.text)
        self.assertEqual(restored.target_attributes, original.target_attributes)
        self.assertEqual(restored.iteration, original.iteration)
        self.assertEqual(restored.attacker_feedback, original.attacker_feedback)
        self.assertEqual(restored.ground_truth, original.ground_truth)

    def test_defender_input_from_dict_defaults(self) -> None:
        result = DefenderInput.from_dict({})
        self.assertEqual(result.text, "")
        self.assertEqual(result.target_attributes, [])
        self.assertEqual(result.iteration, 1)
        self.assertIsNone(result.attacker_feedback)
        self.assertEqual(result.ground_truth, {})

    def test_attacker_output_roundtrip(self) -> None:
        original = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
            raw_response="raw",
        )
        restored = AttackerOutput.from_dict(original.to_dict())
        self.assertEqual(restored.guesses, original.guesses)
        self.assertEqual(restored.reasoning, original.reasoning)
        self.assertEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.successful_attributes, original.successful_attributes)

    def test_strategy_record_roundtrip(self) -> None:
        original = StrategyRecord("Age", "abstraction", "Reason")
        restored = StrategyRecord.from_dict(original.to_dict())
        self.assertEqual(restored.attribute, original.attribute)
        self.assertEqual(restored.strategy, original.strategy)
        self.assertEqual(restored.reasoning, original.reasoning)

    def test_strategy_record_from_dict_defaults(self) -> None:
        result = StrategyRecord.from_dict({})
        self.assertEqual(result.attribute, "")
        self.assertEqual(result.strategy, "omission")
        self.assertEqual(result.reasoning, "")

    def test_adversarial_result_roundtrip(self) -> None:
        iteration = AdversarialIteration(
            iteration=1,
            defender_output=DefenderOutput(
                original_text="orig", rewritten_text="rewritten",
                target_attributes=["Age"],
                strategies_used=[StrategyRecord("Age", "abstraction", "R")],
                confidence=0.9, iteration=1,
            ),
            attacker_output=AttackerOutput(
                guesses={"Age": "46"}, reasoning={"Age": "clue"},
                confidence={"Age": 0.8}, successful_attributes=["Age"],
            ),
            utility_output=UtilityOutput(
                original_text="orig", rewritten_text="rewritten",
                score=0.8, rationale="Good", passes=True,
            ),
            attacker_success=True,
            utility_pass=True,
        )
        result = AdversarialResult(
            final_rewritten_text="rewritten",
            iterations=[iteration],
            total_iterations=1,
            exit_reason="attacker_failed",
            ground_truth={"Age": "46"},
            final_utility_score=0.8,
            success=True,
        )
        restored = AdversarialResult.from_dict(result.to_dict())
        self.assertEqual(restored.final_rewritten_text, result.final_rewritten_text)
        self.assertEqual(restored.total_iterations, 1)
        self.assertEqual(restored.exit_reason, "attacker_failed")
        self.assertTrue(restored.success)
        self.assertEqual(len(restored.iterations), 1)


class TestDefender(unittest.TestCase):
    def test_extract_ground_truth_includes_income(self) -> None:
        defender = Defender(api_key="fake_key")
        defender._call_llm = MagicMock(
            return_value='{"ground_truth": {"income": "$300,000 annually"}}'
        )

        result = defender.extract_ground_truth(
            "Sarah earns $300,000 annually.",
            ["Income"],
        )

        self.assertEqual(result["Income"], "$300,000 annually")
        prompt = defender._call_llm.call_args.args[0][0]
        self.assertIn("Include financial attributes such as income", prompt)
        self.assertIn("earns $300,000 annually", prompt)

    def test_run_does_not_mutate_input_ground_truth(self) -> None:
        defender = Defender(api_key="fake_key")
        defender.extract_ground_truth = MagicMock(return_value={"Age": "46"})
        defender._enumerate_clues = MagicMock(return_value={})
        defender._call_llm = MagicMock(
            return_value="""
            {
              "rewritten_text": "A person is an experienced adult.",
              "strategies_used": [
                {
                  "attribute": "Age",
                  "strategy": "abstraction",
                  "reasoning": "Removed the exact age."
                }
              ],
              "confidence": 0.9
            }
            """
        )
        defender_input = DefenderInput(
            text="Sarah is 46 years old.",
            target_attributes=["Age"],
        )

        result = defender.run(defender_input)

        self.assertEqual(defender_input.ground_truth, {})
        self.assertEqual(result.ground_truth, {"Age": "46"})

    def test_defender_retries_invalid_json(self) -> None:
        defender = Defender(api_key="fake_key")
        valid_response = """
        {
          "rewritten_text": "I remember a major public event with my father.",
          "strategies_used": [
            {
              "attribute": "Age",
              "strategy": "abstraction",
              "reasoning": "Removed the precise age clue."
            }
          ],
          "confidence": 0.9
        }
        """

        defender._call_llm = MagicMock(side_effect=["not json", valid_response])
        result = defender.run(
            DefenderInput(
                text="I was six years old.",
                target_attributes=["Age"],
                iteration=2,
                ground_truth={"Age": "6"},
            )
        )

        self.assertEqual(result.rewritten_text, "I remember a major public event with my father.")
        self.assertEqual(defender._call_llm.call_count, 2)


class TestAttacker(unittest.TestCase):
    @patch("defender.llm_client.genai.Client")
    def test_attacker_run_success(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
          "guesses": {
            "Age": "46",
            "Profession": "Cardiologist"
          },
          "reasoning": {
            "Age": "Mentions medical school timeline",
            "Profession": "Mentions hospital cardiology department"
          },
          "confidence": {
            "Age": 0.8,
            "Profession": 0.9
          },
          "successful_attributes": ["Age", "Profession"]
        }
        """
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        attacker = Attacker(api_key="fake_key")

        output = attacker.run("Rewritten text", ["Age", "Profession"])

        self.assertEqual(output.guesses["Age"], "46")
        self.assertEqual(output.guesses["Profession"], "Cardiologist")
        self.assertEqual(output.confidence["Age"], 0.8)
        self.assertIn("Age", output.successful_attributes)
        self.assertIn("Profession", output.successful_attributes)

    def test_attacker_rejects_empty_input(self) -> None:
        attacker = Attacker(api_key="fake_key")

        with self.assertRaisesRegex(AttackerError, "rewritten_text"):
            attacker.run("   ", ["Age"])

        with self.assertRaisesRegex(AttackerError, "target_attributes"):
            attacker.run("Rewritten text", [])

    def test_attacker_validation_rejects_bad_confidence_and_unknown_success(self) -> None:
        attacker = Attacker(api_key="fake_key")

        errors = attacker._validate_response(
            {
                "guesses": {"Age": "46"},
                "reasoning": {"Age": "Timeline clue"},
                "confidence": {"Age": 1.2},
                "successful_attributes": ["Location"],
            },
            ["Age"],
        )

        joined = " ".join(errors)
        self.assertIn("confidence.Age", joined)
        self.assertIn("unknown attribute", joined)


class TestUtilityJudge(unittest.TestCase):
    def test_score_rejects_empty_texts(self) -> None:
        judge = UtilityJudge(api_key="fake_key")

        with self.assertRaisesRegex(UtilityError, "original_text"):
            judge.score(UtilityInput(original_text=" ", rewritten_text="Rewritten"))

        with self.assertRaisesRegex(UtilityError, "rewritten_text"):
            judge.score(UtilityInput(original_text="Original", rewritten_text=" "))


class TestGeminiClient(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "studio-key",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/service-account.json",
            "GOOGLE_CLOUD_PROJECT": "demo-project",
        },
    )
    @patch("defender.llm_client.os.path.exists", return_value=True)
    @patch("defender.llm_client.genai.Client")
    def test_fallback_switches_to_vertex_after_rate_limit(
        self,
        mock_client_cls: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        class DummyError(Exception):
            """Test-specific Gemini wrapper error."""

        primary = MagicMock()
        fallback = MagicMock()
        primary.models.generate_content.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED"
        )
        fallback_response = MagicMock()
        fallback_response.text = "ok"
        fallback.models.generate_content.return_value = fallback_response
        mock_client_cls.side_effect = [primary, fallback]

        client = GeminiClient(
            model="gemini-test",
            label="test",
            error_type=DummyError,
        )

        self.assertEqual(
            client.generate(["prompt"], "system", 128, 0.0),
            "ok",
        )
        self.assertIs(client._client, fallback)
        self.assertIsNone(client._fallback_client)

        self.assertEqual(
            client.generate(["prompt"], "system", 128, 0.0),
            "ok",
        )
        self.assertEqual(primary.models.generate_content.call_count, 1)
        self.assertEqual(fallback.models.generate_content.call_count, 2)
        mock_exists.assert_called()


class TestOrchestrator(unittest.TestCase):
    def test_run_adversarial_loop_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "text must be non-empty"):
            run_adversarial_loop("", ["Age"])

        with self.assertRaisesRegex(ValueError, "target_attributes"):
            run_adversarial_loop("Text", [])

        with self.assertRaisesRegex(ValueError, "utility_threshold"):
            run_adversarial_loop("Text", ["Age"], utility_threshold=1.5)

    def test_build_attacker_feedback(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46", "Location": "UNKNOWN"},
            reasoning={"Age": "CoT", "Location": "None"},
            confidence={"Age": 0.8, "Location": 0.1},
            successful_attributes=["Age"]
        )
        feedback = _build_attacker_feedback(attacker_output)
        self.assertIn("Guessed '46'", feedback)
        self.assertNotIn("Location", feedback)

    def test_build_attacker_feedback_only_verified_successes(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46", "Profession": "Doctor"},
            reasoning={"Age": "Age clue", "Profession": "Profession clue"},
            confidence={"Age": 0.8, "Profession": 0.9},
            successful_attributes=["Age"],
        )

        feedback = _build_attacker_feedback(attacker_output)

        self.assertIn("Age clue", feedback)
        self.assertNotIn("Profession clue", feedback)

    def test_iteration_feedback_attacker_success_utility_pass(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "Timeline clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
        )

        feedback = _build_iteration_feedback(attacker_output, 0.9, 0.75, True)

        self.assertIn("Timeline clue", feedback)
        self.assertNotIn("utility", feedback.lower())

    def test_iteration_feedback_privacy_success_utility_low(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "UNKNOWN"},
            reasoning={"Age": "No clues"},
            confidence={"Age": 0.1},
            successful_attributes=[],
        )

        feedback = _build_iteration_feedback(attacker_output, 0.4, 0.75, False)

        self.assertIn("Privacy goal achieved", feedback)
        self.assertIn("focus on PRESERVING", feedback)
        self.assertIn("Do NOT apply heavier anonymization", feedback)

    def test_iteration_feedback_attacker_success_utility_low(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "Timeline clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
        )

        feedback = _build_iteration_feedback(attacker_output, 0.4, 0.75, False)

        self.assertIn("The attacker successfully guessed: Age", feedback)
        self.assertIn("Timeline clue", feedback)
        self.assertIn("preserve more non-sensitive narrative detail", feedback)

    def test_rewrite_prompt_uses_lighter_touch_for_utility_recovery(self) -> None:
        prompt = build_rewrite_prompt(
            text="Original text",
            target_attributes=["Age"],
            iteration=2,
            attacker_feedback=(
                "Privacy goal achieved. On the next iteration, focus on PRESERVING "
                "the narrative structure."
            ),
        )

        self.assertIn("achieved privacy but lost too much meaning", prompt)
        self.assertIn("Apply lighter-touch strategies", prompt)
        self.assertNotIn("MUST apply a heavier rewrite", prompt)

    @patch("defender.orchestrator.Defender")
    @patch("defender.orchestrator.Attacker")
    @patch("defender.orchestrator.UtilityJudge")
    def test_run_adversarial_loop_success(
        self,
        mock_judge_cls: MagicMock,
        mock_attacker_cls: MagicMock,
        mock_defender_cls: MagicMock,
    ) -> None:
        # Configure Mocks
        mock_defender = MagicMock()
        mock_attacker = MagicMock()
        mock_judge = MagicMock()
        
        mock_defender_cls.return_value = mock_defender
        mock_attacker_cls.return_value = mock_attacker
        mock_judge_cls.return_value = mock_judge
        
        # Iteration 1 Defender output
        mock_defender.run.return_value = DefenderOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            target_attributes=["Age"],
            strategies_used=[StrategyRecord("Age", "abstraction", "Reason")],
            confidence=0.9,
            iteration=1,
            ground_truth={"Age": "46"}
        )
        
        # Iteration 1 Attacker output (fails to guess, low confidence)
        mock_attacker.run.return_value = AttackerOutput(
            guesses={"Age": "UNKNOWN"},
            reasoning={"Age": "None"},
            confidence={"Age": 0.2},
            successful_attributes=[]
        )
        
        # Iteration 1 Judge output (passing score)
        mock_judge.score.return_value = UtilityOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            score=0.8,
            rationale="Great utility",
            passes=True
        )
        
        result = run_adversarial_loop(
            text="Original text",
            target_attributes=["Age"],
            max_iterations=2
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.exit_reason, "attacker_failed")
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.final_utility_score, 0.8)
        self.assertEqual(result.ground_truth["Age"], "46")

    @patch("defender.orchestrator.Defender")
    @patch("defender.orchestrator.Attacker")
    @patch("defender.orchestrator.UtilityJudge")
    def test_run_adversarial_loop_max_iterations_reached(
        self,
        mock_judge_cls: MagicMock,
        mock_attacker_cls: MagicMock,
        mock_defender_cls: MagicMock,
    ) -> None:
        mock_defender = MagicMock()
        mock_attacker = MagicMock()
        mock_judge = MagicMock()
        mock_defender_cls.return_value = mock_defender
        mock_attacker_cls.return_value = mock_attacker
        mock_judge_cls.return_value = mock_judge

        mock_defender.run.side_effect = [
            DefenderOutput(
                original_text="Original text",
                rewritten_text="Anonymized text 1",
                target_attributes=["Age"],
                strategies_used=[StrategyRecord("Age", "abstraction", "Reason")],
                confidence=0.9,
                iteration=1,
                ground_truth={"Age": "46"},
            ),
            DefenderOutput(
                original_text="Original text",
                rewritten_text="Anonymized text 2",
                target_attributes=["Age"],
                strategies_used=[StrategyRecord("Age", "omission", "Reason")],
                confidence=0.9,
                iteration=2,
                ground_truth={"Age": "46"},
            ),
        ]
        mock_attacker.run.return_value = AttackerOutput(
            guesses={"Age": "mid-40s"},
            reasoning={"Age": "Remaining timeline clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
        )
        mock_judge.score.return_value = UtilityOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            score=0.8,
            rationale="Good utility",
        )

        result = run_adversarial_loop(
            text="Original text",
            target_attributes=["Age"],
            max_iterations=2,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.exit_reason, "max_iterations_reached")
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(result.iterations[-1].attacker_output.successful_attributes, ["Age"])

    @patch("defender.orchestrator.Defender")
    @patch("defender.orchestrator.Attacker")
    @patch("defender.orchestrator.UtilityJudge")
    def test_run_adversarial_loop_uses_custom_confidence_threshold(
        self,
        mock_judge_cls: MagicMock,
        mock_attacker_cls: MagicMock,
        mock_defender_cls: MagicMock,
    ) -> None:
        mock_defender = MagicMock()
        mock_attacker = MagicMock()
        mock_judge = MagicMock()
        mock_defender_cls.return_value = mock_defender
        mock_attacker_cls.return_value = mock_attacker
        mock_judge_cls.return_value = mock_judge

        mock_defender.run.return_value = DefenderOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            target_attributes=["Age"],
            strategies_used=[StrategyRecord("Age", "abstraction", "Reason")],
            confidence=0.9,
            iteration=1,
            ground_truth={"Age": "46"},
        )
        mock_attacker.run.return_value = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "Direct age clue"},
            confidence={"Age": 0.75},
            successful_attributes=["Age"],
        )
        mock_judge.score.return_value = UtilityOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            score=0.8,
            rationale="Good utility",
        )

        result = run_adversarial_loop(
            text="Original text",
            target_attributes=["Age"],
            max_iterations=1,
            confidence_threshold=0.8,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.confidence_threshold, 0.8)
        self.assertEqual(result.iterations[0].attacker_output.successful_attributes, [])
        mock_attacker_cls.assert_called_once()
        self.assertEqual(mock_attacker_cls.call_args.kwargs["confidence_threshold"], 0.8)

    @patch("defender.orchestrator.Defender")
    @patch("defender.orchestrator.Attacker")
    @patch("defender.orchestrator.UtilityJudge")
    def test_run_adversarial_loop_utility_too_low_at_max(
        self,
        mock_judge_cls: MagicMock,
        mock_attacker_cls: MagicMock,
        mock_defender_cls: MagicMock,
    ) -> None:
        mock_defender = MagicMock()
        mock_attacker = MagicMock()
        mock_judge = MagicMock()
        mock_defender_cls.return_value = mock_defender
        mock_attacker_cls.return_value = mock_attacker
        mock_judge_cls.return_value = mock_judge

        mock_defender.run.return_value = DefenderOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            target_attributes=["Age"],
            strategies_used=[StrategyRecord("Age", "abstraction", "Reason")],
            confidence=0.9,
            iteration=1,
            ground_truth={"Age": "46"},
        )
        mock_attacker.run.return_value = AttackerOutput(
            guesses={"Age": "UNKNOWN"},
            reasoning={"Age": "No clues"},
            confidence={"Age": 0.1},
            successful_attributes=[],
        )
        mock_judge.score.return_value = UtilityOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            score=0.4,
            rationale="Too much meaning was lost",
        )

        result = run_adversarial_loop(
            text="Original text",
            target_attributes=["Age"],
            max_iterations=2,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.exit_reason, "utility_too_low_at_max")
        self.assertEqual(result.total_iterations, 2)

    @patch("defender.orchestrator.Defender")
    @patch("defender.orchestrator.Attacker")
    @patch("defender.orchestrator.UtilityJudge")
    def test_ground_truth_is_carried_forward_after_iteration_one(
        self,
        mock_judge_cls: MagicMock,
        mock_attacker_cls: MagicMock,
        mock_defender_cls: MagicMock,
    ) -> None:
        mock_defender = MagicMock()
        mock_attacker = MagicMock()
        mock_judge = MagicMock()
        mock_defender_cls.return_value = mock_defender
        mock_attacker_cls.return_value = mock_attacker
        mock_judge_cls.return_value = mock_judge

        mock_defender.run.side_effect = [
            DefenderOutput(
                original_text="Original text",
                rewritten_text="Anonymized text 1",
                target_attributes=["Age"],
                strategies_used=[StrategyRecord("Age", "abstraction", "Reason")],
                confidence=0.9,
                iteration=1,
                ground_truth={"Age": "46"},
            ),
            DefenderOutput(
                original_text="Original text",
                rewritten_text="Anonymized text 2",
                target_attributes=["Age"],
                strategies_used=[StrategyRecord("Age", "omission", "Reason")],
                confidence=0.9,
                iteration=2,
                ground_truth={"Age": "46"},
            ),
        ]
        mock_attacker.run.side_effect = [
            AttackerOutput(
                guesses={"Age": "46"},
                reasoning={"Age": "Age clue"},
                confidence={"Age": 0.8},
                successful_attributes=["Age"],
            ),
            AttackerOutput(
                guesses={"Age": "UNKNOWN"},
                reasoning={"Age": "No clues"},
                confidence={"Age": 0.1},
                successful_attributes=[],
            ),
        ]
        mock_judge.score.return_value = UtilityOutput(
            original_text="Original text",
            rewritten_text="Anonymized text",
            score=0.8,
            rationale="Good utility",
        )

        run_adversarial_loop(
            text="Original text",
            target_attributes=["Age"],
            max_iterations=2,
        )

        first_input = mock_defender.run.call_args_list[0].args[0]
        second_input = mock_defender.run.call_args_list[1].args[0]
        self.assertEqual(first_input.ground_truth, {})
        self.assertEqual(second_input.ground_truth, {"Age": "46"})


class TestNormalizeClueMap(unittest.TestCase):
    def test_normalizes_known_attributes(self) -> None:
        raw_clue_map = {
            "Age": [
                {"clue": "six years old", "type": "direct", "inference": "states age"}
            ]
        }
        result = _normalize_clue_map(raw_clue_map, ["Age"])
        self.assertEqual(len(result["Age"]), 1)
        self.assertEqual(result["Age"][0]["clue"], "six years old")

    def test_filters_unknown_attributes(self) -> None:
        raw_clue_map = {
            "Age": [{"clue": "clue", "type": "direct", "inference": "inf"}],
            "Secret": [{"clue": "s", "type": "direct", "inference": "i"}],
        }
        result = _normalize_clue_map(raw_clue_map, ["Age"])
        self.assertIn("Age", result)
        self.assertNotIn("Secret", result)

    def test_case_insensitive_lookup(self) -> None:
        raw_clue_map = {
            "age": [{"clue": "clue", "type": "direct", "inference": "inf"}],
        }
        result = _normalize_clue_map(raw_clue_map, ["Age"])
        self.assertEqual(len(result["Age"]), 1)

    def test_non_dict_returns_empty(self) -> None:
        result = _normalize_clue_map("not a dict", ["Age"])
        self.assertEqual(result, {})

    def test_non_list_clues_returns_empty_list(self) -> None:
        result = _normalize_clue_map({"Age": "not a list"}, ["Age"])
        self.assertEqual(result["Age"], [])

    def test_skips_non_dict_clue_entries(self) -> None:
        result = _normalize_clue_map(
            {"Age": [{"clue": "c", "type": "t", "inference": "i"}, "bad"]},
            ["Age"],
        )
        self.assertEqual(len(result["Age"]), 1)


class TestVerifiedSuccessfulAttributes(unittest.TestCase):
    def test_verified_with_ground_truth(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
        )
        result = _verified_successful_attributes(
            attacker_output, ["Age"], {"Age": "46"}, 0.7
        )
        self.assertEqual(result, ["Age"])

    def test_unverified_guess_wrong(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "99"},
            reasoning={"Age": "wrong"},
            confidence={"Age": 0.9},
            successful_attributes=["Age"],
        )
        result = _verified_successful_attributes(
            attacker_output, ["Age"], {"Age": "46"}, 0.7
        )
        self.assertEqual(result, [])

    def test_fallback_to_self_report_without_ground_truth(self) -> None:
        attacker_output = AttackerOutput(
            guesses={"Age": "46"},
            reasoning={"Age": "clue"},
            confidence={"Age": 0.8},
            successful_attributes=["Age"],
        )
        result = _verified_successful_attributes(
            attacker_output, ["Age"], {}, 0.7
        )
        self.assertEqual(result, ["Age"])


class TestDefenderEdgeCases(unittest.TestCase):
    def test_empty_text_raises(self) -> None:
        defender = Defender(api_key="fake_key")
        with self.assertRaisesRegex(DefenderError, "empty"):
            defender.run(DefenderInput(text="  ", target_attributes=["Age"]))

    def test_no_attributes_raises(self) -> None:
        defender = Defender(api_key="fake_key")
        with self.assertRaisesRegex(DefenderError, "attribute"):
            defender.run(DefenderInput(text="Some text", target_attributes=[]))

    def test_text_too_long_raises(self) -> None:
        defender = Defender(api_key="fake_key")
        long_text = "x" * 30000  # ~7500 tokens > 6000 limit
        with self.assertRaisesRegex(DefenderError, "too long"):
            defender.run(DefenderInput(text=long_text, target_attributes=["Age"]))

    def test_extract_ground_truth_empty_attributes(self) -> None:
        defender = Defender(api_key="fake_key")
        result = defender.extract_ground_truth("Some text", [])
        self.assertEqual(result, {})

    def test_extract_ground_truth_handles_exception(self) -> None:
        defender = Defender(api_key="fake_key")
        defender._call_llm = MagicMock(side_effect=RuntimeError("API error"))
        result = defender.extract_ground_truth("Some text", ["Age"])
        self.assertEqual(result, {})


class TestAttackerEdgeCases(unittest.TestCase):
    def test_invalid_confidence_threshold(self) -> None:
        with self.assertRaisesRegex(AttackerError, "confidence_threshold"):
            Attacker(api_key="fake_key", confidence_threshold=1.5)

        with self.assertRaisesRegex(AttackerError, "confidence_threshold"):
            Attacker(api_key="fake_key", confidence_threshold=-0.1)


class TestStrategies(unittest.TestCase):
    def test_valid_strategy_names_matches_enum(self) -> None:
        self.assertEqual(
            VALID_STRATEGY_NAMES,
            {"abstraction", "shifting", "omission"},
        )

    def test_rewrite_strategy_values(self) -> None:
        self.assertEqual(RewriteStrategy.ABSTRACTION.value, "abstraction")
        self.assertEqual(RewriteStrategy.SHIFTING.value, "shifting")
        self.assertEqual(RewriteStrategy.OMISSION.value, "omission")


class TestPrompts(unittest.TestCase):
    def test_build_rewrite_prompt_contains_text_and_attributes(self) -> None:
        prompt = build_rewrite_prompt(
            text="Test text",
            target_attributes=["Age", "Location"],
        )
        self.assertIn("Test text", prompt)
        self.assertIn("Age", prompt)
        self.assertIn("Location", prompt)

    def test_build_rewrite_prompt_iteration_1_no_heavier_rewrite(self) -> None:
        prompt = build_rewrite_prompt(
            text="Test", target_attributes=["Age"], iteration=1
        )
        self.assertNotIn("heavier rewrite", prompt)

    def test_build_rewrite_prompt_iteration_2_heavier_rewrite(self) -> None:
        prompt = build_rewrite_prompt(
            text="Test", target_attributes=["Age"], iteration=2
        )
        self.assertIn("heavier rewrite", prompt)

    def test_build_utility_prompt_no_targets(self) -> None:
        prompt = build_utility_prompt("orig", "rewritten")
        self.assertIn("none provided", prompt)

    def test_build_utility_prompt_with_targets(self) -> None:
        prompt = build_utility_prompt("orig", "rewritten", ["Age"])
        self.assertIn("Age", prompt)
        self.assertNotIn("none provided", prompt)

    def test_build_clue_enumeration_prompt(self) -> None:
        prompt = build_clue_enumeration_prompt("Some text", ["Age"])
        self.assertIn("Some text", prompt)
        self.assertIn("Age", prompt)
        self.assertIn("clue_map", prompt)

    def test_build_ground_truth_prompt(self) -> None:
        prompt = build_ground_truth_prompt("Some text", ["Age"])
        self.assertIn("Some text", prompt)
        self.assertIn("Age", prompt)
        self.assertIn("ground_truth", prompt)

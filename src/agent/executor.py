"""Tool Calling & Execution Engine for Layer 4 with schema validation and retry."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import ValidationError

from src.agent.schemas import EvidencePackage
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient
from src.llm.prompts import build_correction_prompt
from src.llm.schemas import IntentParseResult, ToolCallIntent
from src.tools.registry import (
    TOOL_CATALOGUE,
    execute_tool,
    get_registered_tool_names,
)
from src.tools.schemas import ToolResultModel


class ToolExecutor:
    """
    Executes tool intents parsed by Layer 2 against ingested audio or dataset directories.
    Packages outputs into structured, verifiable evidence with schema error retries.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        max_retries: int = 2,
    ):
        self.client = ollama_client or OllamaClient()
        self.max_retries = max_retries

    def _attempt_argument_correction(
        self,
        tool_name: str,
        invalid_args: Dict[str, Any],
        validation_error: str,
    ) -> Dict[str, Any]:
        """
        Invoke LLM to correct invalid tool arguments when Pydantic constraints are violated.
        """
        tool_info = TOOL_CATALOGUE.get(tool_name, {})
        schema_doc = tool_info.get("args_class", {}).model_json_schema() if tool_info else {}
        
        prompt = (
            f"Tool '{tool_name}' failed parameter validation.\n"
            f"EXPECTED PARAMETER SCHEMA:\n{json.dumps(schema_doc, indent=2)}\n"
            f"INVALID ARGUMENTS PROVIDED:\n{json.dumps(invalid_args, indent=2)}\n"
            f"VALIDATION ERROR:\n{validation_error}\n\n"
            f"Please output ONLY valid JSON matching the parameters dictionary for '{tool_name}'."
        )

        try:
            res_raw = self.client._call_ollama_raw(prompt)
            clean_text = self.client.clean_json_text(res_raw.get("response", "{}"))
            corrected = json.loads(clean_text)
            if isinstance(corrected, dict):
                return corrected
        except Exception:
            pass
        return invalid_args

    def execute_intent(
        self,
        intent_result: IntentParseResult,
        waveform: Optional[np.ndarray] = None,
        sr: Optional[int] = 22050,
        audio_metadata: Optional[AudioMetadata] = None,
        dataset_path: Optional[str] = None,
        query: str = "",
    ) -> EvidencePackage:
        """
        Execute all tool calls in an IntentParseResult and assemble a consolidated EvidencePackage.
        Never raises unhandled exceptions.
        """
        t_start = time.perf_counter()
        audio_id = audio_metadata.file_path if audio_metadata else (dataset_path or "audio_input")

        # 1. Handle explicit abstention
        if intent_result.is_abstention:
            return EvidencePackage(
                query=query,
                audio_id=str(audio_id),
                audio_metadata=audio_metadata,
                status="ABSTAIN",
                abstention_reason=intent_result.abstention_reason or "Question cannot be answered by acoustic analysis.",
                total_dsp_latency_ms=0.0,
            )

        if not intent_result.selected_tools:
            return EvidencePackage(
                query=query,
                audio_id=str(audio_id),
                audio_metadata=audio_metadata,
                status="ABSTAIN",
                abstention_reason="No applicable DSP tools were identified to address the query.",
                total_dsp_latency_ms=0.0,
            )

        # 2. Iterate and execute tool calls
        tool_results: List[ToolResultModel] = []
        tools_called: List[str] = []
        warnings: List[str] = []

        # Propagate Layer 1 ingestion warnings
        if audio_metadata and audio_metadata.warnings:
            warnings.extend(audio_metadata.warnings)

        for tc in intent_result.selected_tools:
            tool_name = tc.tool_name
            tools_called.append(tool_name)
            args_dict = dict(tc.parameters)

            # Special routing for dataset_profiler tool
            if tool_name == "dataset_profiler":
                if dataset_path:
                    args_dict["dataset_path"] = dataset_path
                elif not args_dict.get("dataset_path") or not Path(str(args_dict["dataset_path"])).exists():
                    if audio_metadata and Path(audio_metadata.file_path).is_dir():
                        args_dict["dataset_path"] = audio_metadata.file_path
                    else:
                        args_dict["dataset_path"] = str(audio_id)

            # Pre-validate arguments against tool's Pydantic schema
            tool_entry = TOOL_CATALOGUE.get(tool_name)
            if tool_entry:
                args_class = tool_entry["args_class"]
                for attempt in range(self.max_retries + 1):
                    try:
                        args_class(**args_dict)
                        break
                    except ValidationError as ve:
                        if attempt < self.max_retries:
                            args_dict = self._attempt_argument_correction(
                                tool_name=tool_name,
                                invalid_args=args_dict,
                                validation_error=str(ve),
                            )
                        else:
                            warnings.append(f"Tool '{tool_name}' argument schema validation failed: {str(ve)}")

            # Execute tool deterministically
            res: ToolResultModel = execute_tool(
                name=tool_name,
                waveform=waveform,
                sr=sr or 22050,
                args=args_dict,
            )
            tool_results.append(res)
            if res.warnings:
                warnings.extend(res.warnings)

        # 3. Consolidate evidence
        consolidated_metrics: Dict[str, Any] = {}
        consolidated_anchors: Dict[str, str] = {}
        all_citation_aliases: Dict[str, List[str]] = {}

        success_count = 0
        for r in tool_results:
            if r.status == "success":
                success_count += 1
                prefix = f"{r.tool}." if len(tool_results) > 1 else ""
                for k, v in r.result.items():
                    consolidated_metrics[f"{prefix}{k}"] = v
                for k, v in r.perceptual_anchors.items():
                    consolidated_anchors[f"{prefix}{k}"] = v
                for k, aliases in r.citation_aliases.items():
                    all_citation_aliases[f"{prefix}{k}"] = aliases

        # 4. Determine overall status
        if success_count == len(tool_results):
            overall_status = "SUCCESS"
        elif success_count > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "TOOL_FAILURE"

        total_elapsed = round((time.perf_counter() - t_start) * 1000, 2)

        return EvidencePackage(
            query=query,
            audio_id=str(audio_id),
            audio_metadata=audio_metadata,
            status=overall_status,
            tools_called=tools_called,
            tool_results=tool_results,
            consolidated_metrics=consolidated_metrics,
            consolidated_anchors=consolidated_anchors,
            all_citation_aliases=all_citation_aliases,
            total_dsp_latency_ms=total_elapsed,
            warnings=warnings,
        )


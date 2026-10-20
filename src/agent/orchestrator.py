"""Layer 4: AuralQ Agent Orchestration Loop and Session Coordinator."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.executor import ToolExecutor
from src.agent.schemas import AgentResponse, EvidencePackage, TurnRecord
from src.agent.state import SessionState
from src.agent.synthesizer import ResponseSynthesizer
from src.agent.validator import CitationValidator
from src.audio.loader import AudioLoadResult, load_audio_file
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient
from src.llm.schemas import ToolCallIntent
from src.tools.schemas import ToolResultModel


class AuralQAgent:
    """
    Top-level orchestrator for AuralQ.
    Coordinates audio ingestion, intent parsing, deterministic tool execution,
    tool result caching, multi-turn state, grounded natural-language synthesis,
    and deterministic numeric citation verification.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        executor: Optional[ToolExecutor] = None,
        synthesizer: Optional[ResponseSynthesizer] = None,
        session_state: Optional[SessionState] = None,
        validator: Optional[CitationValidator] = None,
    ):
        self.client = client or OllamaClient()
        self.executor = executor or ToolExecutor(client=self.client)
        self.synthesizer = synthesizer or ResponseSynthesizer(client=self.client)
        self.session_state = session_state or SessionState()
        self.validator = validator or CitationValidator()

    def load_audio(self, file_path_or_data: Union[str, Path, Any]) -> AudioLoadResult:
        """
        Ingest and validate an untrusted audio file through Layer 1 hygiene pipeline.
        Blocks DC bias, gates silence, inspects clipping, and resets the tool cache.
        """
        load_res = load_audio_file(file_path_or_data)
        if load_res.status == "SUCCESS" and load_res.metadata is not None:
            self.session_state.set_audio(
                metadata=load_res.metadata,
                waveform=load_res.waveform,
                sr=load_res.sample_rate,
            )
        return load_res

    def load_dataset(self, dataset_path: Union[str, Path]) -> str:
        """
        Validate and register an audio dataset directory for profiling.
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Dataset path must be a directory: {path}")

        resolved_str = str(path.resolve())
        self.session_state.set_dataset(resolved_str)
        return resolved_str

    def query(
        self,
        question: str,
        audio_path: Optional[Union[str, Path]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> AgentResponse:
        """
        Execute a complete end-to-end conversational turn.
        1. Resolve active inputs (audio/dataset).
        2. Parse intent via Layer 2 LLM.
        3. Dispatch deterministic DSP tools with cache reuse.
        4. Synthesize natural-language answer grounded in evidence.
        5. Record turn in session history.
        """
        t_total_start = time.perf_counter()

        # Step 0: Auto-detect existing directory in query if no inputs provided
        if not dataset_path and not audio_path:
            for token in question.split():
                clean_tok = token.strip("`'\".,:;()")
                if ("/" in clean_tok or "\\" in clean_tok) and Path(clean_tok).exists() and Path(clean_tok).is_dir():
                    dataset_path = clean_tok
                    break

        # Step 1: Handle input overrides if passed
        if audio_path:
            load_res = self.load_audio(audio_path)
            if load_res.status != "SUCCESS":
                evidence = EvidencePackage(
                    query=question,
                    status="ABSTAIN" if load_res.status == "ABSTAIN" else "TOOL_FAILURE",
                    abstention_reason=load_res.message,
                    warnings=[load_res.message],
                )
                turn = TurnRecord(
                    turn_id=len(self.session_state.turns) + 1,
                    timestamp=time.time(),
                    query=question,
                    evidence=evidence,
                    response_text=f"Audio ingestion status ({load_res.status}): {load_res.message}",
                    latency_breakdown={"total_ms": round((time.perf_counter() - t_total_start) * 1000, 2)},
                )
                self.session_state.add_turn(turn)
                return AgentResponse(
                    query=question,
                    response_text=turn.response_text,
                    status="abstained" if load_res.status == "ABSTAIN" else "dsp_error",
                    evidence=evidence,
                    turn_record=turn,
                    latency_breakdown=turn.latency_breakdown,
                )

        if dataset_path:
            self.load_dataset(dataset_path)

        # Step 2: Intent Parsing via Layer 2 LLM
        meta_dict = None
        if self.session_state.audio_metadata:
            meta_dict = self.session_state.audio_metadata.to_context_dict()

        t_intent_start = time.perf_counter()
        active_ds = self.session_state.dataset_path
        llm_res = self.client.parse_intent(
            question,
            audio_metadata=meta_dict,
            dataset_path=active_ds,
        )
        intent_ms = round((time.perf_counter() - t_intent_start) * 1000, 2)

        if not llm_res.is_valid or llm_res.parsed_intent is None:
            err_msg = llm_res.error_message or "Unknown LLM intent parsing error"

            # Format friendly, actionable explanation if 429 (quota) or 503 (service unavailable) occurs
            is_quota_error = any(indicator in err_msg for indicator in ("429", "Too Many Requests", "RESOURCE_EXHAUSTED", "quota"))
            is_service_unavailable = any(indicator in err_msg for indicator in ("503", "Service Unavailable", "502", "504", "overloaded"))

            if is_quota_error:
                response_text = (
                    "⚠️ **Hosted Cloud API Quota Exhausted (HTTP 429: Too Many Requests)**\n\n"
                    "The shared rate limit or quota for the hosted cloud API key has been temporarily reached.\n\n"
                    "**How you can continue:**\n"
                    "1. 🔒 **Switch to the Local App (Recommended)**: Run AuralQ locally on your computer using Ollama for "
                    "**unlimited queries, permanent session history, and 100% air-gapped privacy** with zero API keys. "
                    "Simply double-click `run_auralq.bat` (or `./run_auralq.sh` on macOS/Linux).\n"
                    "2. 🔑 **Use Your Own API Key**: Enter your personal free Gemini or Groq API key in the sidebar under "
                    "**Cloud Demo Engine** to continue querying in this web session."
                )
            elif is_service_unavailable:
                response_text = (
                    "⚠️ **Cloud AI Service Temporarily Overloaded (HTTP 503: Service Unavailable)**\n\n"
                    "Google's cloud AI servers are currently experiencing high traffic, transient model overload, or temporary downtime.\n\n"
                    "**How you can continue immediately:**\n"
                    "1. 🔒 **Switch to Local Mode (Recommended)**: Run AuralQ locally on your machine with Ollama for **100% offline, zero-downtime, and private execution**. "
                    "Simply double-click `run_auralq.bat` (or `./run_auralq.sh` on macOS/Linux).\n"
                    "2. ⚡ **Switch Provider or Retry**: In the sidebar under **Cloud Demo Engine**, try switching from Gemini to Groq (or vice-versa), or retry your query in a few moments."
                )
            else:
                response_text = f"I was unable to interpret the acoustic intent for this query: {err_msg}"

            evidence = EvidencePackage(
                query=question,
                status="TOOL_FAILURE",
                warnings=[err_msg],
            )
            turn = TurnRecord(
                turn_id=len(self.session_state.turns) + 1,
                timestamp=time.time(),
                query=question,
                evidence=evidence,
                response_text=response_text,
                latency_breakdown={"intent_ms": intent_ms, "total_ms": round((time.perf_counter() - t_total_start) * 1000, 2)},
            )
            self.session_state.add_turn(turn)
            return AgentResponse(
                query=question,
                response_text=turn.response_text,
                status=llm_res.status,  # "connection_error" or "schema_error"
                evidence=evidence,
                turn_record=turn,
                latency_breakdown=turn.latency_breakdown,
            )

        intent = llm_res.parsed_intent

        # Dataset mode enforcement & single-track safeguard:
        q_lower = question.lower()
        is_dataset_query = (
            "dataset" in q_lower
            or "directory" in q_lower
            or ("audit" in q_lower and "file" not in q_lower and "track" not in q_lower)
        )
        if active_ds and (is_dataset_query or not self.session_state.waveform):
            intent.is_abstention = False
            has_profiler = any(tc.tool_name == "dataset_profiler" for tc in intent.selected_tools)
            if not has_profiler:
                intent.selected_tools = [ToolCallIntent(
                    tool_name="dataset_profiler",
                    parameters={"dataset_path": active_ds},
                )]

        # Safeguard: if dataset_profiler was selected but no dataset is registered
        # and a single audio waveform is active, replace with single-file diagnostic tools
        resolved_tools = []
        for tc in intent.selected_tools:
            if tc.tool_name == "dataset_profiler":
                ds_candidate = (
                    self.session_state.dataset_path
                    or tc.parameters.get("dataset_path")
                    or (self.session_state.audio_metadata.file_path if (self.session_state.audio_metadata and Path(self.session_state.audio_metadata.file_path).is_dir()) else None)
                )
                if not ds_candidate and self.session_state.waveform is not None:
                    # Single audio file active: convert to single-track noise & quality diagnostics
                    resolved_tools.append(ToolCallIntent(tool_name="spectral_flatness", parameters={}))
                    resolved_tools.append(ToolCallIntent(tool_name="rms_energy", parameters={}))
                    resolved_tools.append(ToolCallIntent(tool_name="snr_estimate", parameters={}))
                    continue
            resolved_tools.append(tc)
        intent.selected_tools = resolved_tools[:3]

        # Step 3: Fast-path for Abstention
        if intent.is_abstention:
            evidence = EvidencePackage(
                query=question,
                status="ABSTAIN",
                abstention_reason=intent.abstention_reason,
            )
            resp_text, synth_ms = self.synthesizer.synthesize(question, evidence)
            grounding_report = self.validator.validate(resp_text, evidence, question)
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
            latencies = {
                "intent_ms": intent_ms,
                "dsp_ms": 0.0,
                "synthesis_ms": synth_ms,
                "validation_ms": 0.0,
                "total_ms": total_ms,
            }
            turn = TurnRecord(
                turn_id=len(self.session_state.turns) + 1,
                timestamp=time.time(),
                query=question,
                parsed_intent=intent,
                evidence=evidence,
                response_text=resp_text,
                latency_breakdown=latencies,
                grounding_report=grounding_report,
            )
            self.session_state.add_turn(turn)
            return AgentResponse(
                query=question,
                response_text=resp_text,
                status="abstained",
                evidence=evidence,
                turn_record=turn,
                latency_breakdown=latencies,
                grounding_report=grounding_report,
            )

        # Step 4: Tool Execution with Cache Interception
        t_dsp_start = time.perf_counter()
        tools_called: List[str] = []
        tool_results: List[ToolResultModel] = []
        cached_tools_used: List[str] = []
        warnings: List[str] = []

        if self.session_state.audio_metadata and self.session_state.audio_metadata.warnings:
            warnings.extend(self.session_state.audio_metadata.warnings)

        for tc in intent.selected_tools:
            tool_name = tc.tool_name
            tools_called.append(tool_name)
            args_dict = dict(tc.parameters)

            # Route dataset_profiler parameter
            if tool_name == "dataset_profiler":
                if self.session_state.dataset_path:
                    args_dict["dataset_path"] = self.session_state.dataset_path
                elif not args_dict.get("dataset_path") or not Path(str(args_dict["dataset_path"])).exists():
                    if self.session_state.audio_metadata and Path(self.session_state.audio_metadata.file_path).is_dir():
                        args_dict["dataset_path"] = self.session_state.audio_metadata.file_path
                if not args_dict.get("dataset_path") or not Path(str(args_dict["dataset_path"])).is_dir():
                    # No directory path available for dataset profiling; skip execution to prevent TypeError
                    continue

            # Check cache
            cached_res = self.session_state.get_cached_tool_result(tool_name, args_dict)
            if cached_res is not None:
                cached_tools_used.append(tool_name)
                tool_results.append(cached_res)
                if cached_res.warnings:
                    warnings.extend(cached_res.warnings)
            else:
                # Execute tool via ToolExecutor
                fresh_res = self.executor.execute_tool(
                    tool_name=tool_name,
                    waveform=self.session_state.waveform,
                    sr=self.session_state.sr,
                    parameters=args_dict,
                )
                tool_results.append(fresh_res)
                if fresh_res.warnings:
                    warnings.extend(fresh_res.warnings)

                # Store in session tool cache if successful
                if fresh_res.status == "success":
                    self.session_state.cache_tool_result(tool_name, args_dict, fresh_res)

        # If all dispatched tools were resolved from cache, DSP compute time is 0.0 ms
        if cached_tools_used and len(cached_tools_used) == len(intent.selected_tools):
            dsp_ms = 0.0
        else:
            dsp_ms = round((time.perf_counter() - t_dsp_start) * 1000, 2)

        # Step 5: Consolidate Metrics, Anchors, and Citations into EvidencePackage
        consolidated_metrics: Dict[str, Any] = {}
        consolidated_anchors: Dict[str, str] = {}
        all_citation_aliases: Dict[str, List[str]] = {}

        has_failures = any(r.status != "success" for r in tool_results)
        all_failed = all(r.status != "success" for r in tool_results) if tool_results else True

        if not tool_results or all_failed:
            overall_status = "TOOL_FAILURE"
        elif has_failures:
            overall_status = "PARTIAL"
        else:
            overall_status = "SUCCESS"

        for r in tool_results:
            if r.status == "success":
                for k, v in r.result.items():
                    consolidated_metrics[f"{r.tool}.{k}"] = v
                for k, v in r.perceptual_anchors.items():
                    consolidated_anchors[f"{r.tool}.{k}"] = v
                for k, aliases in r.citation_aliases.items():
                    all_citation_aliases[f"{r.tool}.{k}"] = aliases

        evidence = EvidencePackage(
            query=question,
            audio_id=self.session_state.audio_id,
            audio_metadata=self.session_state.audio_metadata,
            status=overall_status,
            tools_called=tools_called,
            tool_results=tool_results,
            consolidated_metrics=consolidated_metrics,
            consolidated_anchors=consolidated_anchors,
            all_citation_aliases=all_citation_aliases,
            total_dsp_latency_ms=dsp_ms,
            warnings=list(dict.fromkeys(warnings)),
        )

        # Step 6: Response Synthesis
        history_ctx = self.session_state.get_recent_history_context(max_turns=3)
        raw_resp_text, synth_ms = self.synthesizer.synthesize(
            query=question,
            evidence=evidence,
            history_context=history_ctx,
        )

        # Step 7: Deterministic Citation Validation & Grounding Repair
        t_val_start = time.perf_counter()
        final_text, grounding_report = self.validator.repair_if_unverified(
            response_text=raw_resp_text,
            evidence=evidence,
            query=question,
            synthesizer=self.synthesizer,
        )
        val_ms = round((time.perf_counter() - t_val_start) * 1000, 2)

        total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
        latencies = {
            "intent_ms": intent_ms,
            "dsp_ms": dsp_ms,
            "synthesis_ms": synth_ms,
            "validation_ms": val_ms,
            "total_ms": total_ms,
        }

        turn = TurnRecord(
            turn_id=len(self.session_state.turns) + 1,
            timestamp=time.time(),
            query=question,
            parsed_intent=intent,
            evidence=evidence,
            response_text=final_text,
            latency_breakdown=latencies,
            cached_tools_used=cached_tools_used,
            grounding_report=grounding_report,
        )
        self.session_state.add_turn(turn)

        return AgentResponse(
            query=question,
            response_text=final_text,
            status="success" if overall_status in ["SUCCESS", "PARTIAL"] else "dsp_error",
            evidence=evidence,
            turn_record=turn,
            latency_breakdown=latencies,
            cached_tools_used=cached_tools_used,
            grounding_report=grounding_report,
        )

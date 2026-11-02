"""Integration Test: Frontier Multi-Agent Autonomous Swarm Roleplay.

Simulates a 4-agent collaborative system:
1. Orchestrator / Planner: Decomposes enterprise architecture goals.
2. Research Agent: Queries knowledge base (simulating loop detection on repeated failure).
3. Code Generation Agent: Emits truncated/malformed JSON schema under token limit.
4. Security Auditor Agent: Inspects outputs for PII leaks and prompt injections.

Verifies:
- TrajectoryTracer recording and cost attribution
- LoopDetector catching cyclic tool calls
- Schema repairing corrupted agent JSON payloads
- Guardrails auditing agent communications
- AgentGraph generating mermaid visualization
- TrajectoryReplayer verifying deterministic playback
"""
import json

from veriflux.trajectory.tracer import TrajectoryTracer
from veriflux.trajectory.loop_detector import LoopDetector
from veriflux.trajectory.graph import AgentGraph
from veriflux.trajectory.replayer import TrajectoryReplayer
from veriflux.schema.json_repair import JSONRepair
from veriflux.schema.validator import SchemaValidator
from veriflux.guardrails.engine import GuardrailEngine
from veriflux.tokens.budget_manager import TokenBudgetManager


def test_frontier_multiagent_swarm_simulation():
    # Setup Swarm Monitoring
    tracer = TrajectoryTracer("swarm_session_001")
    loop_detector = LoopDetector(window_size=4, max_repeats=3)
    graph = AgentGraph()
    budget = TokenBudgetManager(max_tokens=100_000, max_cost_usd=5.0)
    guard = GuardrailEngine(pii_action="block")
    repairer = JSONRepair()
    validator = SchemaValidator()

    # Step 1: Orchestrator decomposes goal
    tracer.record_step(
        agent="Orchestrator",
        action="decompose_goal",
        input_data={"goal": "Build JWT authentication microservice with rate limiting"},
        output_data={
            "subtasks": ["Research existing auth tokens", "Generate auth code", "Security review"]
        },
        tokens=120,
        latency_ms=45.0,
    )
    graph.add_transition("User", "Orchestrator")
    graph.add_transition("Orchestrator", "Researcher")

    # Step 2: Researcher queries knowledge base, falls into repeated loop trap
    query = "SELECT token_schema FROM auth_specs WHERE version = 'v2'"
    loop_caught = False

    for attempt in range(5):
        tracer.record_step(
            agent="Researcher",
            action="query_db",
            input_data={"query": query},
            output_data={"error": "Connection timeout on db_replica_2"},
            tokens=45,
            latency_ms=120.0,
        )
        is_loop = loop_detector.check_step("query_db", query)
        if is_loop:
            loop_caught = True
            # Swarm recovers from loop trap by routing to fallback docs
            tracer.record_step(
                agent="Researcher",
                action="fallback_cached_spec",
                input_data={"source": "cache"},
                output_data={"spec": "JWT HS256 with user_id, roles, exp"},
                tokens=80,
                latency_ms=15.0,
            )
            break

    assert loop_caught, "Loop detector should catch repeated identical database query failures"
    graph.add_transition("Researcher", "Coder")

    # Step 3: Coder generates JSON configuration but output is truncated mid-way
    truncated_llm_json = '{"service": "auth_service", "algorithm": "HS256", "rate_limit": {"window_sec": 60, "max_requests": 100, "headers": ["X-RateLimit-Limit", "X-RateLimit-Remaining"'
    
    # Veriflux heals the mangled JSON output in real-time
    repaired_json_str = repairer.repair(truncated_llm_json)
    parsed_config = json.loads(repaired_json_str)

    assert parsed_config["service"] == "auth_service"
    assert parsed_config["algorithm"] == "HS256"
    assert "headers" in parsed_config["rate_limit"]
    assert isinstance(parsed_config["rate_limit"]["headers"], list)

    tracer.record_step(
        agent="Coder",
        action="generate_config",
        input_data={"spec": "auth_spec"},
        output_data=parsed_config,
        tokens=350,
        latency_ms=210.0,
    )
    graph.add_transition("Coder", "Auditor")

    # Step 4: Security Auditor inspects code comments and commits for secrets
    proposed_commit_msg = "Implemented JWT auth. Testing with sample dev token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID"
    audit_eval = guard.check_input(proposed_commit_msg)

    # Veriflux catches sensitive token
    assert not audit_eval.passed, "Auditor should flag PII / JWT token leak"
    assert audit_eval.details["pii"]["count"] > 0

    redacted_msg = guard.pii_guard.redact(proposed_commit_msg)
    assert "eyJ" not in redacted_msg
    assert "[JWT]" in redacted_msg

    tracer.record_step(
        agent="Auditor",
        action="redact_and_approve",
        input_data={"raw_commit": proposed_commit_msg},
        output_data={"sanitized_commit": redacted_msg, "status": "APPROVED"},
        tokens=90,
        latency_ms=30.0,
    )
    graph.add_transition("Auditor", "User")

    # Step 5: Verify Trajectory Analysis & Replay
    summary = tracer.analyze()
    assert summary.total_steps >= 5
    assert summary.total_tokens > 500

    # Mermaid graph rendering
    mermaid_diag = graph.to_mermaid()
    assert "graph TD" in mermaid_diag or "flowchart" in mermaid_diag
    assert "Orchestrator" in mermaid_diag
    assert "Researcher" in mermaid_diag
    assert "Coder" in mermaid_diag
    assert "Auditor" in mermaid_diag

    # Trajectory Replayer verification
    replayer = TrajectoryReplayer(tracer.steps)
    replayed_steps = []
    for step in replayer.replay():
        replayed_steps.append(step.agent)

    assert "Orchestrator" in replayed_steps
    assert "Researcher" in replayed_steps
    assert "Coder" in replayed_steps
    assert "Auditor" in replayed_steps

"""AgentAudit — fault injection and reliability measurement for agent pipelines."""

from .agent import ExtractionResult, SelfCorrectingAgent
from .corpus import export_corpus, load_corpus, verify_corpus
from .dataset import build_corpus, corpus_for_injector
from .faults.injector import FaultInjector, InjectionPlan, ToolFailure
from .faults.taxonomy import ALL_FAULTS, TAXONOMY, Detectability, FaultLayer
from .judge import SemanticJudge
from .metrics import Report, RunScore, format_table, score_run
from .providers.base import Completion, Provider
from .providers.mock import MockProvider
from .runner import Condition, Experiment, ablation_conditions, standard_conditions
from .schema import (
    INVOICE_CHECKS, INVOICE_SCHEMA, Criticality, Field, FieldType, Schema,
)
from .tools.base import Tool, ToolRegistry
from .tools.extraction import DEFAULT_TOOLS
from .trace import Span, SpanKind, SpanStatus, Trace

__version__ = "0.1.0"

__all__ = [
    "SelfCorrectingAgent", "ExtractionResult",
    "InjectionPlan", "FaultInjector", "ToolFailure",
    "TAXONOMY", "ALL_FAULTS", "FaultLayer", "Detectability",
    "Schema", "Field", "FieldType", "Criticality",
    "INVOICE_SCHEMA", "INVOICE_CHECKS",
    "Trace", "Span", "SpanKind", "SpanStatus",
    "Provider", "Completion", "MockProvider",
    "Tool", "ToolRegistry", "DEFAULT_TOOLS",
    "Report", "RunScore", "score_run", "format_table",
    "Experiment", "Condition", "standard_conditions", "ablation_conditions",
    "SemanticJudge", "build_corpus", "corpus_for_injector",
    "load_corpus", "export_corpus", "verify_corpus",
    "__version__",
]

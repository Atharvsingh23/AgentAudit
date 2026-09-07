"""AgentAudit — fault injection and reliability measurement for agent pipelines."""

from .agent import ExtractionResult, SelfCorrectingAgent
from .corpus import export_corpus, load_corpus, verify_corpus
from .dataset import build_corpus, corpus_for_injector
from .faults.injector import (
    NO_FAULTS, FaultContext, FaultInjector, InjectionPlan, ToolFailure,
)
from .faults.taxonomy import (
    ALL_FAULTS, TAXONOMY, Detectability, FaultLayer, FaultSpec,
)
from .judge import Judgement, SemanticJudge
from .metrics import FieldScore, Report, RunScore, format_table, score_run
from .providers.base import Completion, Provider
from .providers.mock import MockProvider
from .runner import (
    Condition, Experiment, ExperimentResult, ablation_conditions,
    by_fault_conditions, standard_conditions,
)
from .schema import (
    INVOICE_CHECKS, INVOICE_SCHEMA, ConsistencyCheck, Criticality, Field,
    FieldType, Schema,
)
from .tools.base import Tool, ToolRegistry
from .tools.extraction import DEFAULT_TOOLS
from .trace import Span, SpanKind, SpanStatus, Trace

__version__ = "0.1.0"

__all__ = [
    "SelfCorrectingAgent", "ExtractionResult",
    "InjectionPlan", "FaultInjector", "FaultContext", "ToolFailure", "NO_FAULTS",
    "TAXONOMY", "ALL_FAULTS", "FaultLayer", "Detectability", "FaultSpec",
    "Schema", "Field", "FieldType", "Criticality", "ConsistencyCheck",
    "INVOICE_SCHEMA", "INVOICE_CHECKS",
    "Trace", "Span", "SpanKind", "SpanStatus",
    "Provider", "Completion", "MockProvider",
    "Tool", "ToolRegistry", "DEFAULT_TOOLS",
    "Report", "RunScore", "FieldScore", "score_run", "format_table",
    "Experiment", "ExperimentResult", "Condition",
    "standard_conditions", "ablation_conditions", "by_fault_conditions",
    "SemanticJudge", "Judgement", "build_corpus", "corpus_for_injector",
    "load_corpus", "export_corpus", "verify_corpus",
    "__version__",
]

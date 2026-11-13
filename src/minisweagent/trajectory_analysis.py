import json
import sqlite3
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel


class EvidenceRef(BaseModel):
    trajectory_id: str
    source: Literal["initial_input", "command", "output", "submission"]
    action_id: int | None
    note: str


class Comparison(BaseModel):
    id: str
    stage: str
    success_behavior: str
    failure_behavior: str
    interpretation: str
    evidence_refs: list[EvidenceRef]
    uncertainties: list[str]


class Lesson(BaseModel):
    id: str
    title: str
    scope: Literal["general", "domain"]
    applicability: str
    trigger: str
    recommended_steps: list[str]
    avoid: list[str]
    rationale: str
    comparison_ids: list[str]
    evidence_refs: list[EvidenceRef]
    boundaries: list[str]
    confidence: Literal["low", "medium", "high"]
    uncertainties: list[str]


class Analysis(BaseModel):
    task_summary: str
    comparisons: list[Comparison]
    lessons: list[Lesson]
    limitations: list[str]


class ReadActionOutput(BaseModel):
    trajectory_id: str
    action_id: int


TOOLS = [
    {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema.model_json_schema()},
    }
    for name, description, schema in [
        ("read_action_output", "Read the complete stored output of one historical action.", ReadActionOutput),
        ("submit_analysis", "Submit the final structured analysis and finish. Call this tool alone.", Analysis),
    ]
]


class AnalysisModel(LitellmModel):
    def _query(self, messages: list[dict], **kwargs):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=TOOLS,
            **(self.config.model_kwargs | kwargs),
        )

    def _parse_actions(self, response) -> list[dict]:
        return [
            {
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "tool_call_id": call.id,
            }
            for call in response.choices[0].message.tool_calls or []
        ]


class AnalysisEnvironment:
    def __init__(self, db: Path):
        self.db = db.resolve()

    def execute(self, action: dict, cwd: str = "") -> dict:
        if action["name"] == "submit_analysis":
            submission = Analysis.model_validate(action["arguments"]).model_dump_json()
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        if action["name"] != "read_action_output":
            raise ValueError(f"Unknown tool: {action['name']}")
        args = ReadActionOutput.model_validate(action["arguments"])
        connection = sqlite3.connect(f"{self.db.as_uri()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT output_json FROM outputs WHERE trajectory_id = ? AND action_id = ?",
                (args.trajectory_id, args.action_id),
            ).fetchone()
        finally:
            connection.close()
        return {
            "output": json.dumps(
                {
                    **args.model_dump(),
                    "found": row is not None,
                    **({"output": json.loads(row[0])} if row is not None else {}),
                }
            ),
            "returncode": 0,
            "exception_info": "",
        }

    def get_template_vars(self, **kwargs) -> dict:
        return kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": {"db": str(self.db)}}}}

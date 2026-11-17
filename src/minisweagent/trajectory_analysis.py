import json
import sqlite3
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel, Field

from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


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
    comparisons: list[Comparison] = Field(
        description="Complete evidence-backed comparisons with no count limit, independent of lesson selection."
    )
    lessons: list[Lesson] = Field(description="Selected case lessons from the completed comparisons; may be empty.")
    limitations: list[str]


class ReadSteps(BaseModel):
    trajectory_id: str
    action_ids: list[int] = Field(min_length=1)


class ReadActionOutput(BaseModel):
    trajectory_id: str
    action_id: int


def get_tools(max_lessons: int = 3) -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": schema.model_json_schema()},
        }
        for name, description, schema in [
            ("read_steps", "Re-read complete original steps by action ID, without their outputs.", ReadSteps),
            ("read_action_output", "Read the complete stored output of one historical action.", ReadActionOutput),
            ("submit_analysis", "Submit the final structured analysis and finish. Call this tool alone.", Analysis),
        ]
    ]
    for tool in tools:
        if tool["function"]["name"] == "submit_analysis":
            tool["function"]["parameters"]["properties"]["lessons"]["maxItems"] = max_lessons
    return tools


class AnalysisModelConfig(LitellmModelConfig):
    max_lessons: int = 3


class AnalysisModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=AnalysisModelConfig, **kwargs)

    def _query(self, messages: list[dict], **kwargs):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=get_tools(self.config.max_lessons),
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
    def __init__(self, db: Path, max_lessons: int = 3, *, trajectories: list[dict] | None = None):
        self.db = db.resolve()
        self.max_lessons = max_lessons
        self.steps: dict[str, dict[int, dict]] = {}
        for trajectory in trajectories or []:
            trajectory_id = trajectory["trajectory_id"]
            if trajectory_id in self.steps:
                raise ValueError(f"Duplicate trajectory ID: {trajectory_id}")
            self.steps[trajectory_id] = {}
            for step in trajectory["turns"]:
                if step["id"] in self.steps[trajectory_id]:
                    raise ValueError(f"Duplicate action ID {step['id']} in trajectory {trajectory_id}")
                self.steps[trajectory_id][step["id"]] = step

    def execute(self, action: dict, cwd: str = "") -> dict:
        if action["name"] == "read_steps":
            args = ReadSteps.model_validate(action["arguments"])
            steps = self.steps.get(args.trajectory_id, {})
            return {
                "output": json.dumps(
                    {
                        "trajectory_id": args.trajectory_id,
                        "results": [
                            {
                                "action_id": action_id,
                                "found": action_id in steps,
                                **({"step": steps[action_id]} if action_id in steps else {}),
                            }
                            for action_id in args.action_ids
                        ],
                    }
                ),
                "returncode": 0,
                "exception_info": "",
            }
        if action["name"] == "submit_analysis":
            analysis = Analysis.model_validate(action["arguments"])
            if len(analysis.lessons) > self.max_lessons:
                raise ValueError(f"Expected at most {self.max_lessons} lessons, got {len(analysis.lessons)}")
            submission = analysis.model_dump_json()
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
        return {"max_lessons": self.max_lessons, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": {"db": str(self.db), "max_lessons": self.max_lessons}}}}

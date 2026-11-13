import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.analyze_trajectories import run_pairs
from minisweagent.trajectory_analysis import TOOLS, AnalysisEnvironment, AnalysisModel


@pytest.fixture
def data(tmp_path):
    db = tmp_path / "outputs.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE outputs (trajectory_id TEXT, action_id INTEGER, output_json TEXT)")
        connection.executemany(
            "INSERT INTO outputs VALUES (?, ?, ?)",
            [
                ("success", 1, json.dumps("success output")),
                ("failure", 1, json.dumps("failure output")),
            ],
        )
    for label in ["success", "failure"]:
        (tmp_path / f"{label}.json").write_text(
            json.dumps(
                {
                    "trajectory_id": label,
                    "initial_input": [{"role": "user", "content": "Fix the issue"}],
                    "turns": [{"id": 1, "name": "inspect", "arguments": {"path": "code.py"}}],
                    "submission": "patch",
                }
            )
        )
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            [{"task_id": task, "success": "success.json", "failure": "failure.json"} for task in ["first", "second"]]
        )
    )
    return pairs, db


def test_read_outputs(data):
    env = AnalysisEnvironment(data[1])
    for trajectory in ["success", "failure"]:
        assert json.loads(
            env.execute(
                {
                    "name": "read_action_output",
                    "arguments": {"trajectory_id": trajectory, "action_id": 1},
                }
            )["output"]
        ) == {
            "trajectory_id": trajectory,
            "action_id": 1,
            "found": True,
            "output": f"{trajectory} output",
        }
    assert json.loads(
        env.execute(
            {
                "name": "read_action_output",
                "arguments": {"trajectory_id": "success", "action_id": 2},
            }
        )["output"]
    ) == {"trajectory_id": "success", "action_id": 2, "found": False}
    with sqlite3.connect(data[1]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outputs").fetchone() == (2,)


def test_parse_tools_and_observations():
    model = AnalysisModel(model_name="test")
    call = SimpleNamespace(
        id="read_1",
        function=SimpleNamespace(
            name="read_action_output",
            arguments='{"trajectory_id": "success", "action_id": 1}',
        ),
    )
    actions = model._parse_actions(
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
    )
    assert actions == [
        {
            "name": "read_action_output",
            "arguments": {"trajectory_id": "success", "action_id": 1},
            "tool_call_id": "read_1",
        }
    ]
    observation = model.format_observation_messages(
        {"extra": {"actions": actions}},
        [{"output": "evidence", "returncode": 0, "exception_info": ""}],
    )[0]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "read_1"
    assert "evidence" in observation["content"]
    assert [tool["function"]["name"] for tool in TOOLS] == ["read_action_output", "submit_analysis"]
    assert "lessons" in TOOLS[1]["function"]["parameters"]["required"]


@pytest.mark.parametrize(
    ("step_limit", "cost_limit", "status"),
    [(50, 3, "Submitted"), (1, 3, "LimitsExceeded"), (50, 1, "LimitsExceeded")],
)
def test_batch(data, tmp_path, step_limit, cost_limit, status, reset_global_stats):
    settings = get_config_from_spec(builtin_config_dir / "trajectory_analysis.yaml")
    settings["agent"]["step_limit"] = step_limit
    settings["agent"]["cost_limit"] = cost_limit
    settings["model"] |= {
        "model_name": "deterministic",
        "outputs": [
            make_toolcall_output(
                "Read evidence",
                [],
                [
                    {
                        "name": "read_action_output",
                        "arguments": {"trajectory_id": "success", "action_id": 1},
                        "tool_call_id": "read_1",
                    }
                ],
            ),
            make_toolcall_output(
                "Submit",
                [],
                [
                    {
                        "name": "submit_analysis",
                        "arguments": {
                            "task_summary": "A candidate analysis",
                            "comparisons": [],
                            "lessons": [],
                            "limitations": [],
                        },
                        "tool_call_id": "submit_1",
                    }
                ],
            ),
        ],
    }
    output = tmp_path / "results"
    output.mkdir()
    (output / "first.analysis.json").write_text("stale")
    run_pairs(*data, output, settings, model_class=DeterministicToolcallModel)
    for task in ["first", "second"]:
        trajectory = json.loads((output / f"{task}.traj.json").read_text())
        assert trajectory["info"]["exit_status"] == status
        assert trajectory["info"]["model_stats"]["api_calls"] == (2 if status == "Submitted" else 1)
        assert "success output" in trajectory["messages"][3]["content"]
        assert '"name": "inspect"' in trajectory["messages"][1]["content"]
        if status == "Submitted":
            report = json.loads((output / f"{task}.analysis.json").read_text())
            assert report["pair"] == {
                "task_id": task,
                "success_trajectory_id": "success",
                "failure_trajectory_id": "failure",
                "label_source": "human",
            }
            assert report["lessons"] == [] and report["schema_version"] == "1.0"
        else:
            assert not (output / f"{task}.analysis.json").exists()


def test_invalid_submission(data):
    with pytest.raises(ValidationError):
        AnalysisEnvironment(data[1]).execute({"name": "submit_analysis", "arguments": {"task_summary": "Incomplete"}})

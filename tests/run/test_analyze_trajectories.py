import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.exceptions import Submitted
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.analyze_trajectories import run_pairs
from minisweagent.trajectory_analysis import AnalysisEnvironment, AnalysisModel, get_tools


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
    tools = {tool["function"]["name"]: tool["function"] for tool in get_tools()}
    assert set(tools) == {"read_steps", "read_action_output", "submit_analysis"}
    assert tools["read_steps"]["parameters"]["properties"]["action_ids"]["minItems"] == 1
    assert "lessons" in tools["submit_analysis"]["parameters"]["required"]
    assert tools["submit_analysis"]["parameters"]["properties"]["lessons"]["maxItems"] == 3
    assert AnalysisModel(model_name="test").config.max_lessons == 3


@pytest.mark.parametrize(
    ("step_limit", "cost_limit", "status"),
    [(50, 3, "Submitted"), (1, 3, "LimitsExceeded"), (50, 1, "LimitsExceeded")],
)
def test_batch(data, tmp_path, step_limit, cost_limit, status, reset_global_stats):
    settings = get_config_from_spec(builtin_config_dir / "trajectory_analysis.yaml")
    settings["agent"]["step_limit"] = step_limit
    settings["agent"]["cost_limit"] = cost_limit
    settings["max_lessons"] = 2
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
        assert "Extract at most 2 concrete" in trajectory["messages"][0]["content"]
        assert trajectory["info"]["config"]["environment"]["max_lessons"] == 2
        if status == "Submitted":
            report = json.loads((output / f"{task}.analysis.json").read_text())
            assert report["pair"] == {
                "task_id": task,
                "success_trajectory_id": "success",
                "failure_trajectory_id": "failure",
                "label_source": "human",
            }
            assert report["lessons"] == [] and report["schema_version"] == "1.1"
            assert report["analysis_spec_version"] == "1.3"
        else:
            assert not (output / f"{task}.analysis.json").exists()


def test_invalid_submission(data):
    with pytest.raises(ValidationError):
        AnalysisEnvironment(data[1]).execute({"name": "submit_analysis", "arguments": {"task_summary": "Incomplete"}})


@pytest.mark.parametrize(("max_lessons", "count"), [(3, 0), (3, 3), (3, 4), (1, 1), (1, 2)])
def test_lesson_limit(data, max_lessons, count):
    evidence = {"trajectory_id": "success", "source": "output", "action_id": 1, "note": "Observed output"}
    lesson = {
        "id": "L1",
        "title": "Preserve local axis state",
        "applicability": "Unshared subplot axes",
        "trigger": "Different category sets",
        "recommended_steps": ["Read each axis's ticks"],
        "avoid": ["Using the global category count"],
        "rationale": "Each subplot has its own categories",
        "comparison_ids": ["C1"],
        "evidence_refs": [evidence],
        "boundaries": ["Unshared axes"],
        "confidence": "medium",
        "uncertainties": ["Failure not reproduced"],
    }
    arguments = {
        "task_summary": "Nominal axis limits",
        "comparisons": [
            {
                "id": f"C{i}",
                "stage": f"Finalization check {i}",
                "success_behavior": "Uses axis ticks",
                "failure_behavior": "Uses global categories",
                "interpretation": "May lose local context",
                "evidence_refs": [evidence],
                "uncertainties": ["Not reproduced"],
            }
            for i in range(5)
        ],
        "lessons": [lesson | {"id": f"L{i}"} for i in range(count)],
        "limitations": [],
    }
    schema = next(
        tool["function"]["parameters"]
        for tool in get_tools(max_lessons)
        if tool["function"]["name"] == "submit_analysis"
    )
    assert schema["properties"]["lessons"]["maxItems"] == max_lessons
    assert "maxItems" not in schema["properties"]["comparisons"]
    assert "scope" not in schema["$defs"]["Lesson"]["properties"]
    assert AnalysisModel(model_name="test", max_lessons=max_lessons).config.max_lessons == max_lessons
    env = AnalysisEnvironment(data[1], max_lessons=max_lessons)
    if count > max_lessons:
        with pytest.raises(ValueError, match=f"Expected at most {max_lessons} lessons"):
            env.execute({"name": "submit_analysis", "arguments": arguments})
    else:
        with pytest.raises(Submitted) as submitted:
            env.execute({"name": "submit_analysis", "arguments": arguments})
        assert json.loads(submitted.value.messages[0]["extra"]["submission"]) == arguments


def test_default_config():
    settings = get_config_from_spec(builtin_config_dir / "trajectory_analysis.yaml")
    assert settings["max_lessons"] == 3
    assert settings["analysis_spec_version"] == "1.3"


@pytest.mark.parametrize(
    ("trajectory_id", "action_ids"),
    [
        ("success", [15]),
        ("success", [42, 15, 42]),
        ("success", [42, 99, 15]),
        ("success", [99]),
        ("unknown", [15]),
    ],
)
def test_read_steps(tmp_path, trajectory_id, action_ids):
    steps = [
        {"id": 15, "command": "echo original", "metadata": {"note": "preserve"}},
        {"id": 42, "name": "inspect", "arguments": {"path": "code.py"}},
    ]
    db = tmp_path / "absent.sqlite"
    env = AnalysisEnvironment(db, trajectories=[{"trajectory_id": "success", "turns": steps}])
    output = env.execute(
        {
            "name": "read_steps",
            "arguments": {
                "trajectory_id": trajectory_id,
                "action_ids": action_ids,
            },
        }
    )
    expected = {step["id"]: step for step in steps} if trajectory_id == "success" else {}
    assert json.loads(output["output"]) == {
        "trajectory_id": trajectory_id,
        "results": [
            {
                "action_id": action_id,
                "found": action_id in expected,
                **({"step": expected[action_id]} if action_id in expected else {}),
            }
            for action_id in action_ids
        ],
    }
    assert output["returncode"] == 0 and output["exception_info"] == ""
    assert not db.exists()
    assert env.serialize()["info"]["config"]["environment"] == {"db": str(db), "max_lessons": 3}


def test_read_steps_validation_and_legacy_environment(tmp_path):
    env = AnalysisEnvironment(tmp_path / "absent.sqlite")
    assert json.loads(
        env.execute(
            {
                "name": "read_steps",
                "arguments": {
                    "trajectory_id": "unknown",
                    "action_ids": [1],
                },
            }
        )["output"]
    )["results"] == [{"action_id": 1, "found": False}]
    for arguments in [
        {"trajectory_id": "success", "action_ids": []},
        {"trajectory_id": "success", "action_ids": ["invalid"]},
        {"trajectory_id": "success"},
    ]:
        with pytest.raises(ValidationError):
            env.execute({"name": "read_steps", "arguments": arguments})


@pytest.mark.parametrize(
    ("trajectories", "error"),
    [
        ([{"trajectory_id": "same", "turns": []}, {"trajectory_id": "same", "turns": []}], "Duplicate trajectory ID"),
        ([{"trajectory_id": "same", "turns": [{"id": 7}, {"id": 7}]}], "Duplicate action ID"),
    ],
)
def test_duplicate_step_indexes(tmp_path, trajectories, error):
    with pytest.raises(ValueError, match=error):
        AnalysisEnvironment(tmp_path / "absent.sqlite", trajectories=trajectories)


def test_reread_output_and_submit(data, tmp_path, reset_global_stats):
    settings = get_config_from_spec(builtin_config_dir / "trajectory_analysis.yaml")
    calls = [
        ("read_steps", {"trajectory_id": "success", "action_ids": [1]}, "steps_1"),
        ("read_action_output", {"trajectory_id": "success", "action_id": 1}, "output_1"),
        (
            "submit_analysis",
            {"task_summary": "Checked", "comparisons": [], "lessons": [], "limitations": []},
            "submit_1",
        ),
    ]
    settings["model"] |= {
        "model_name": "deterministic",
        "outputs": [
            make_toolcall_output("Analyze", [], [{"name": name, "arguments": arguments, "tool_call_id": call_id}])
            for name, arguments, call_id in calls
        ],
    }
    run_pairs(*data, tmp_path / "results", settings, model_class=DeterministicToolcallModel)
    for task in ["first", "second"]:
        trajectory = json.loads((tmp_path / "results" / f"{task}.traj.json").read_text())
        messages = trajectory["messages"]
        initial = json.loads(messages[1]["content"].split("\n", 1)[1])
        for label in ["success", "failure"]:
            assert initial[label] == json.loads((data[0].parent / f"{label}.json").read_text())
        assert "read_steps" in messages[0]["content"]
        assert messages[3]["role"] == "tool" and messages[3]["tool_call_id"] == "steps_1"
        assert json.loads(messages[3]["content"])["results"][0]["step"] == initial["success"]["turns"][0]
        assert messages[5]["tool_call_id"] == "output_1"
        assert json.loads(messages[5]["content"])["output"] == "success output"
        assert trajectory["info"]["model_stats"]["api_calls"] == 3
        assert trajectory["info"]["exit_status"] == "Submitted"
        report = json.loads((tmp_path / "results" / f"{task}.analysis.json").read_text())
        assert report["task_summary"] == "Checked" and report["schema_version"] == "1.1"
        assert report["analysis_spec_version"] == "1.3"

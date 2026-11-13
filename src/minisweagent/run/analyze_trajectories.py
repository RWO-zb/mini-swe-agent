"""Compare human-labeled trajectory pairs and extract transferable lessons."""

import json
from pathlib import Path

import typer

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model_name
from minisweagent.trajectory_analysis import AnalysisEnvironment, AnalysisModel

app = typer.Typer()


@app.command()
def main(
    pairs: Path = typer.Option(..., "--pairs"),
    db: Path = typer.Option(..., "--db"),
    model: str | None = typer.Option(None, "--model"),
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    config: Path = typer.Option(builtin_config_dir / "trajectory_analysis.yaml", "--config"),
) -> None:
    settings = get_config_from_spec(config)
    run_pairs(pairs, db, output_dir, settings, model)


def run_pairs(
    pairs: Path, db: Path, output_dir: Path, settings: dict, model: str | None = None, *, model_class=AnalysisModel
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in json.loads(pairs.read_text()):
        success = json.loads((pairs.parent / pair["success"]).read_text())
        failure = json.loads((pairs.parent / pair["failure"]).read_text())
        report_path = output_dir / f"{pair['task_id']}.analysis.json"
        report_path.unlink(missing_ok=True)
        agent = DefaultAgent(
            model_class(**(settings.get("model", {}) | {"model_name": get_model_name(model, settings.get("model"))})),
            AnalysisEnvironment(db),
            **(settings["agent"] | {"output_path": output_dir / f"{pair['task_id']}.traj.json"}),
        )
        result = agent.run(
            json.dumps(
                {
                    "task_id": pair["task_id"],
                    "label_source": "human",
                    "success": success,
                    "failure": failure,
                }
            )
        )
        if result["exit_status"] == "Submitted":
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "analysis_spec_version": settings["analysis_spec_version"],
                        "pair": {
                            "task_id": pair["task_id"],
                            "success_trajectory_id": success["trajectory_id"],
                            "failure_trajectory_id": failure["trajectory_id"],
                            "label_source": "human",
                        },
                        **json.loads(result["submission"]),
                    },
                    indent=2,
                )
            )
        typer.echo(f"{pair['task_id']}: {result['exit_status']}")


if __name__ == "__main__":
    app()

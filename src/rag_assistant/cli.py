"""Command line: rag generate | ingest | ask | eval | demo | slack | feedback."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from rag_assistant import corpus, evaluation
from rag_assistant.app import build_assistant, build_index
from rag_assistant.config import Settings
from rag_assistant.feedback import FeedbackStore
from rag_assistant.llm import FakeGenerator, make_generator
from rag_assistant.search import make_embedder

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Policy Q&A with citations, refusals and evaluation.",
)
console = Console()


@app.command()
def generate() -> None:
    """Write the synthetic policy corpus to corpus/."""
    settings = Settings.from_env()
    console.print(
        f"Wrote {corpus.write_corpus(settings.corpus_dir)} documents to {settings.corpus_dir}"
    )


@app.command()
def ingest() -> None:
    """Parse, chunk, embed and index the corpus. Unchanged documents are skipped."""
    c = build_index(Settings.from_env())
    r = c.report
    console.print(
        f"Index ({c.embedder.name}): {r.added} added, {r.updated} updated, "
        f"{r.removed} removed, {r.unchanged} unchanged; {r.chunks} chunks"
    )


def _print_answer(a: Any) -> None:
    console.print(f"[bold]Q:[/bold] {a.question}")
    console.print(f"[bold]A:[/bold] {a.text}")
    for n in a.cited:
        chunk = a.sources[n - 1].chunk
        tag = " [yellow](superseded)[/yellow]" if chunk.superseded else ""
        console.print(f"   [{n}] {chunk.title} > {chunk.section}{tag}")
    console.print(f"   [dim]evidence {a.evidence:.2f}, {a.reason}[/dim]\n")


@app.command()
def ask(question: str) -> None:
    """Answer one question from the command line."""
    _print_answer(build_assistant(Settings.from_env()).ask(question))


@app.command()
def demo() -> None:
    """Index the corpus and answer a few example questions, offline."""
    settings = Settings.from_env()
    corpus.write_corpus(settings.corpus_dir)
    assistant = build_assistant(settings)
    for q in (
        "How long do I have to ask for a refund?",
        "I left my laptop on the train, how quickly do I need to tell someone?",
        "How many days of annual leave do full-time staff get?",
        "Can I bring my dog to the office?",
    ):
        _print_answer(assistant.ask(q))


CONFIGS = [("bm25", "section"), ("dense", "section"), ("hybrid", "section"), ("hybrid", "fixed")]


@app.command("eval")
def eval_cmd(
    rerank: Annotated[
        bool, typer.Option(help="Add a cross-encoder row (embeddings extra)")
    ] = False,
    sweep: Annotated[bool, typer.Option(help="Print the evidence-threshold sweep")] = False,
    gate: Annotated[bool, typer.Option(help="Exit 1 if below eval/baseline.json")] = False,
    update_baseline: Annotated[bool, typer.Option(help="Write eval/baseline.json")] = False,
) -> None:
    """Retrieval comparison, generation metrics and refusal behaviour on the golden set."""
    settings = Settings.from_env()
    questions = evaluation.load_golden(settings.golden_path)
    embedder = make_embedder(settings.embedder)
    results: dict[str, Any] = {"embedder": embedder.name, "retrieval": {}}

    table = Table(title=f"Retrieval on {embedder.name} embeddings (50 answerable questions)")
    for col in (
        "Method",
        "Chunking",
        "Hit@1",
        "Hit@3",
        "MRR",
        "Section Hit@1",
        "Paraphrase Hit@1",
        "Superseded Hit@1",
    ):
        table.add_column(col)
    reranked = [("dense", "section", "cross-encoder"), ("hybrid", "section", "cross-encoder")]
    configs = [*CONFIGS, *reranked] if rerank else CONFIGS
    components = {}
    for cfg in configs:
        method, chunking = cfg[0], cfg[1]
        reranker = cfg[2] if len(cfg) > 2 else "none"
        if chunking not in components:
            components[chunking] = build_index(settings, embedder, chunking)
        assistant = build_assistant(settings, components[chunking], method, reranker)
        r = evaluation.evaluate_retrieval(assistant.retriever, questions)
        name = method + (" + rerank" if reranker != "none" else "")
        results["retrieval"][f"{name}/{chunking}"] = r
        o, bt = r["overall"], r["by_type"]
        table.add_row(
            name,
            chunking,
            f"{o['hit@1']:.2f}",
            f"{o['hit@3']:.2f}",
            f"{o['mrr']:.2f}",
            f"{r['section']['hit@1']:.2f}",
            f"{bt['paraphrase']['hit@1']:.2f}",
            f"{bt['superseded']['hit@1']:.2f}",
        )
    console.print(table)

    default = build_assistant(
        settings, components.get(settings.chunking) or build_index(settings, embedder)
    )
    judge = (
        FakeGenerator()
        if settings.llm_provider == "fake"
        else make_generator(settings.llm_provider, settings.answer_model)
    )
    gen = evaluation.evaluate_generation(default, judge, questions)
    g = gen.as_dict()
    results["generation"] = g
    results["config"] = {
        "retrieval": settings.retrieval,
        "chunking": settings.chunking,
        "threshold": settings.threshold,
        "llm": settings.llm_provider,
    }
    console.print(
        f"Generation ({settings.retrieval}/{settings.chunking}, threshold "
        f"{settings.threshold}, {settings.llm_provider} generator): answer accuracy "
        f"{g['answer_accuracy']:.0%} on 50 answerable; unanswerable refused "
        f"{g['refusal_accuracy_unanswerable']:.0%}; false refusals "
        f"{g['false_refusal_rate_answerable']:.0%}; citation validity "
        f"{g['citation_validity']:.0%}; groundedness {g['groundedness']:.0%}; superseded "
        f"questions answered from the current policy {g['superseded_used_current']:.0%}"
    )
    by_type = Table(title="Answers by question type")
    for col in ("Type", "Questions", "Correct", "Refused"):
        by_type.add_column(col)
    for t, s in g["by_type"].items():
        by_type.add_row(
            t, str(s["n"]), "-" if t == "unanswerable" else str(s["correct"]), str(s["refused"])
        )
    console.print(by_type)

    if sweep:
        rows = evaluation.sweep(default, questions, [round(0.05 * i, 2) for i in range(2, 17)])
        results["sweep"] = rows
        st = Table(title="Evidence threshold sweep")
        for col in ("Threshold", "Unanswerable refused", "Answerable kept"):
            st.add_column(col)
        for row in rows:
            st.add_row(
                f"{row['threshold']:.2f}",
                f"{row['unanswerable_refused']:.0%}",
                f"{row['answerable_kept']:.0%}",
            )
        console.print(st)

    out = settings.baseline_path.parent / "results" / f"latest-{embedder.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**results, "rows": gen.rows}, indent=2) + "\n", encoding="utf-8")
    baseline_key = embedder.name + ("+rerank" if settings.reranker != "none" else "")
    if update_baseline:
        existing = (
            json.loads(settings.baseline_path.read_text("utf-8"))
            if settings.baseline_path.is_file()
            else {}
        )
        existing[baseline_key] = {
            "hybrid_section_mrr": results["retrieval"]["hybrid/section"]["overall"]["mrr"],
            "answer_accuracy": g["answer_accuracy"],
            "refusal_accuracy_unanswerable": g["refusal_accuracy_unanswerable"],
            "citation_validity": g["citation_validity"],
        }
        settings.baseline_path.write_text(json.dumps(existing, indent=2) + "\n", "utf-8")
        console.print(f"Baseline written to {settings.baseline_path}")
    if gate:
        base = json.loads(settings.baseline_path.read_text("utf-8"))[baseline_key]
        current = {
            "hybrid_section_mrr": results["retrieval"]["hybrid/section"]["overall"]["mrr"],
            **{
                k: g[k]
                for k in ("answer_accuracy", "refusal_accuracy_unanswerable", "citation_validity")
            },
        }
        failed = [
            f"{k} {current[k]:.3f} < baseline {v:.3f} - 0.02"
            for k, v in base.items()
            if current[k] < v - 0.02
        ]
        for f in failed:
            console.print(f"[red]Gate:[/red] {f}")
        console.print("[green]Gate passed[/green]" if not failed else "[red]Gate failed[/red]")
        if failed:
            raise typer.Exit(1)


@app.command()
def feedback() -> None:
    """Show helpful / not helpful counts and recent unhelpful answers."""
    settings = Settings.from_env()
    console.print_json(json.dumps(FeedbackStore(settings.state_dir / "feedback.sqlite").summary()))


@app.command()
def slack() -> None:
    """Run the Slack app in Socket Mode (needs SLACK_BOT_TOKEN and SLACK_APP_TOKEN)."""
    settings = Settings.from_env()
    if not settings.slack_bot_token or not settings.slack_app_token:
        console.print("[red]Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN (see README).[/red]")
        raise typer.Exit(2)
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    from rag_assistant.slack_app import create_app

    bolt = create_app(
        build_assistant(settings),
        FeedbackStore(settings.state_dir / "feedback.sqlite"),
        settings.slack_bot_token,
    )
    SocketModeHandler(bolt, settings.slack_app_token).start()  # type: ignore[no-untyped-call]


@app.callback()
def _main() -> None:
    _ = replace  # keep import for settings overrides in scripts


if __name__ == "__main__":
    app()

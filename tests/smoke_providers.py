"""Live smoke test for every configured LLM provider (not collected by pytest).

Run manually:  python tests/smoke_providers.py

For each provider with a key in .env it checks, in order:
  1. model discovery (lists the models the key can use)
  2. plain chat
  3. tool-calling (the deep agent depends on it)
  4. a tiny burst to observe rate-limit behaviour
Then it checks that a DeepAgents subagent's /evidence files are visible to the parent.
"""
from __future__ import annotations

import os
import time
import traceback

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# Overridable via env so a retired model name never needs a code edit.
PROVIDERS = {
    "gemini": {
        "key": "GOOGLE_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default": "gemini-2.5-flash",
    },
    "groq": {
        "key": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default": "nvidia/nemotron-3-super-120b-a12b:free",
    },
}


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def make_model(provider: str, model: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def list_models(provider: str) -> list[str]:
    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return [m.name for m in client.models.list()]
    if provider == "groq":
        from groq import Groq

        return [m.id for m in Groq().models.list().data]
    import httpx

    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"] if m["id"].endswith(":free")]


def check(label: str, fn) -> bool:
    start = time.perf_counter()
    try:
        detail = fn()
        print(f"  PASS  {label:<14} {time.perf_counter() - start:5.1f}s  {detail or ''}")
        return True
    except Exception as exc:  # smoke test: report, never crash
        print(f"  FAIL  {label:<14} {type(exc).__name__}: {str(exc)[:200]}")
        return False


def smoke_provider(provider: str, cfg: dict) -> dict[str, bool]:
    model_name = os.getenv(cfg["model_env"], cfg["default"])
    print(f"\n[{provider}] model={model_name}")
    llm = make_model(provider, model_name)
    results: dict[str, bool] = {}

    def discovery():
        models = list_models(provider)
        found = any(model_name in m for m in models)
        sample = ", ".join(models[:6])
        return f"{len(models)} models; configured model {'found' if found else 'NOT FOUND'}; e.g. {sample}"

    def chat():
        return llm.invoke("Reply with the single word: ok").content

    def tool_call():
        msg = llm.bind_tools([add]).invoke("What is 17 + 25? Use the add tool.")
        calls = msg.tool_calls
        assert calls, "no tool_calls returned"
        assert calls[0]["name"] == "add" and {calls[0]["args"]["a"], calls[0]["args"]["b"]} == {17, 25}, calls
        return f"args={calls[0]['args']}"

    def burst():
        ok = 0
        for _ in range(5):
            llm.invoke("Say hi")
            ok += 1
        return f"{ok}/5 rapid calls succeeded"

    results["discovery"] = check("discovery", discovery)
    results["chat"] = check("chat", chat)
    results["tool_calling"] = check("tool_calling", tool_call)
    results["burst"] = check("burst x5", burst)
    return results


def smoke_deepagents(provider: str, cfg: dict) -> bool:
    """Do files a subagent writes to the virtual FS reach its parent?"""
    print(f"\n[deepagents on {provider}]")

    def run():
        from deepagents import create_deep_agent

        llm = make_model(provider, os.getenv(cfg["model_env"], cfg["default"]))
        agent = create_deep_agent(
            model=llm,
            system_prompt=(
                "Delegate to the 'writer' subagent: ask it to write the text 'evidence-ok' to "
                "/evidence/x.md. Then read /evidence/x.md yourself and reply with its exact contents."
            ),
            subagents=[
                {
                    "name": "writer",
                    "description": "Writes a given text to a given file path.",
                    "system_prompt": "Use write_file to save exactly the text you are asked to save.",
                }
            ],
        )
        out = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        files = out.get("files", {})
        assert "/evidence/x.md" in files, f"file not in shared state: {list(files)}"
        return f"files={list(files)}"

    return check("shared FS", run)


def main() -> None:
    summary: dict[str, dict[str, bool]] = {}
    for provider, cfg in PROVIDERS.items():
        if not os.getenv(cfg["key"]):
            print(f"\n[{provider}] SKIPPED ({cfg['key']} not set)")
            continue
        try:
            summary[provider] = smoke_provider(provider, cfg)
        except Exception:
            traceback.print_exc()
            summary[provider] = {"setup": False}

    # DeepAgents check runs on the first provider whose tool-calling passed.
    for provider, res in summary.items():
        if res.get("tool_calling"):
            summary[provider]["deepagents_fs"] = smoke_deepagents(provider, PROVIDERS[provider])
            break

    print("\n=== summary ===")
    for provider, res in summary.items():
        print(f"{provider:<11}", "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in res.items()))
    if not summary:
        print("No provider keys found. Fill in .env first.")


if __name__ == "__main__":
    main()

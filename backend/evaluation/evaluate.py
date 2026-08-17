import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Add the root directory to sys.path so Python can find the 'backend' package
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Force-load the .env file from the project root directory
from dotenv import load_dotenv
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

import pandas as pd
import mlflow
from mlflow.metrics import EvaluationMetric, MetricValue
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from backend.tools import parser, writer, executor

# Resolve relative path to the dataset file
DATA_PATH = os.path.join(current_dir, "evaluation_dataset.json")


def run_single_query(ticker: str, timeframe: str, action: str) -> dict:
    """Mirrors the retry loop inside executor.generate_chart, but returns
    the generated code and raw execution result instead of MCP-formatted
    tool output, since that's what the judge needs to grade."""
    parsed = parser.parse_query_params(ticker, timeframe, action)
    if not parsed["valid"]:
        return {
            "generated_code": "No code generated.",
            "execution_result": f"Invalid request: {parsed['error']}",
        }

    tickers = parsed["tickers"]
    data_path = parsed["data_path"]
    error = ""
    code = ""
    result = None

    for attempt in range(1, executor.MAX_RETRIES + 1):
        code = writer.write_code(tickers, data_path, action, error)
        result = executor.run_python(code, executor.OUTPUT_DIR)

        if result["success"]:
            chart_path = next(
                (p for p in result["artifacts"] if p.endswith(".png")), None
            )
            if chart_path:
                return {
                    "generated_code": code,
                    "execution_result": (
                        f"Success on attempt {attempt}. "
                        f"stdout:\n{result['stdout']}\n"
                        f"Chart saved to: {chart_path}"
                    ),
                }
            error = (
                "The script ran without errors but did NOT save any .png "
                "file to OUTPUT_DIR."
            )
            continue

        error = result["error"]

    return {
        "generated_code": code,
        "execution_result": f"Failed after {executor.MAX_RETRIES} attempts. Last error:\n{error}",
    }


def run_agent_pipeline(model_input):
    """
    Expects model_input with columns: ticker, timeframe, action
    (there's no NL-routing agent anymore, so these must be supplied
    directly rather than as one free-text query).
    """
    tickers = model_input["ticker"].tolist()
    timeframes = model_input["timeframe"].tolist()
    actions = model_input["action"].tolist()

    predictions = []
    for ticker, timeframe, action in zip(tickers, timeframes, actions):
        print(f"\nRunning test case: ticker={ticker}, timeframe={timeframe}, action={action}")

        outcome = run_single_query(ticker, timeframe, action)
        code = outcome["generated_code"]
        execution = outcome["execution_result"]

        combined_output = f"--- GENERATED CODE ---\n{code}\n\n--- EXECUTION RESULT ---\n{execution}"
        predictions.append(combined_output)

    return predictions


def ollama_judge_fn(predictions, targets, metrics, **kwargs):
    # Ornith (DeepReinforce), served locally via Ollama, as the grading engine
    judge_llm = ChatOllama(model="ornith", temperature=0.0)

    judge_prompt = PromptTemplate.from_template(
        "You are an expert quantitative code reviewer.\n"
        "Evaluate the following agent output based strictly on this rubric:\n{rubric}\n\n"
        "Agent Output:\n{output}\n\n"
        "Provide a score from 1 to 5 (where 5 is perfect), followed by a short justification.\n"
        "Format your response exactly like this:\n"
        "SCORE: <number>\n"
        "JUSTIFICATION: <text>"
    )

    scores = []
    justifications = []

    for pred, rubric in zip(predictions, targets):
        prompt_text = judge_prompt.format(rubric=rubric, output=pred)
        response = judge_llm.invoke(prompt_text).content

        try:
            score_line = [line for line in response.split('\n') if 'SCORE:' in line.upper()][0]
            score = float(score_line.split(':')[1].strip())
            justification = response.split('JUSTIFICATION:')[1].strip()
        except Exception:
            score = 1.0
            justification = f"Parsing failed. Raw judge output: {response}"

        scores.append(score)
        justifications.append(justification)

    return MetricValue(
        scores=scores,
        justifications=justifications,
        aggregate_results={"mean_score": sum(scores) / len(scores)}
    )


ollama_metric = EvaluationMetric(
    name="ollama_code_judge",
    greater_is_better=True,
    eval_fn=ollama_judge_fn
)


if __name__ == "__main__":
    # Load the ground truth dataset
    df_eval = pd.read_json(DATA_PATH)

    # Start MLflow Evaluation
    with mlflow.start_run(run_name="ollama_pipeline_eval"):
        print(f"Loaded {len(df_eval)} test cases. Starting tool pipeline execution...")

        results = mlflow.evaluate(
            data=df_eval,
            model=run_agent_pipeline,
            model_type="text",
            targets="rubric",
            extra_metrics=[ollama_metric]
        )

        print("\nEvaluation Complete!")
        print(f"Average Judge Score: {results.metrics['ollama_code_judge/mean_score']:.2f}/5.00")
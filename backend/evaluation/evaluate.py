import os
import sys
from dotenv import load_dotenv


current_dir = os.path.dirname(__file__)

# Step up two levels to the root directory (C:\Market Intelligence Agent)
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the root directory to sys.path so Python can find the 'backend' package
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Force-load the .env file from the project root directory
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# ==========================================
# Standard Imports
# ==========================================
import pandas as pd
import mlflow
from mlflow.metrics import EvaluationMetric, MetricValue
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from backend.agents.graph import financial_analyst_graph

# Resolve relative path to the dataset file
DATA_PATH = os.path.join(current_dir, "evaluation_dataset.json")

def run_agent_pipeline(model_input):
    queries = model_input["inputs"].tolist()
    
    predictions = []
    for query in queries:
        print(f"\nRunning test query: {query}")
        
        # Invoke using the verified 'query' key
        final_state = financial_analyst_graph.invoke({"query": query})
        
        code = final_state.get("generated_code", "No code generated.")
        execution = final_state.get("execution_result") or final_state.get("error_traceback", "Unknown execution state.")
        
        combined_output = f"--- GENERATED CODE ---\n{code}\n\n--- EXECUTION RESULT ---\n{execution}"
        predictions.append(combined_output)
        
    return predictions

# ==========================================
# 2. Custom NVIDIA Judge Metric
# ==========================================
def nvidia_judge_fn(predictions, targets, metrics, **kwargs):
    # Initializing NVIDIA's Llama 3.1 70B model as the grading engine
    judge_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.0)
    
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

nvidia_metric = EvaluationMetric(
    name="nvidia_code_judge",
    greater_is_better=True,
    eval_fn=nvidia_judge_fn
)

# ==========================================
# 3. Execution Block
# ==========================================
if __name__ == "__main__":
    # Load the ground truth dataset
    df_eval = pd.read_json(DATA_PATH)
    
    # Start MLflow Evaluation
    with mlflow.start_run(run_name="nvidia_langgraph_eval"):
        print(f"Loaded {len(df_eval)} test cases. Starting LangGraph pipeline execution...")
        
        # Note: Depending on your exact MLflow version, you might see a deprecation warning 
        # regarding mlflow.evaluate. It acts as a fallback and will still safely execute. 
        results = mlflow.evaluate(
            data=df_eval,
            model=run_agent_pipeline,
            model_type="text",
            targets="rubric", 
            extra_metrics=[nvidia_metric] 
        )
        
        print("\nEvaluation Complete!")
        print(f"Average Judge Score: {results.metrics['nvidia_code_judge/mean_score']:.2f}/5.00")
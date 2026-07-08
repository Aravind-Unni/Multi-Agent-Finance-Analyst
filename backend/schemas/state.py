from typing import TypedDict, Optional, Dict, Any

class AgentState(TypedDict):
    query: str                          
    parsed_params: Optional[Dict[str, Any]] 
    generated_code: Optional[str]       
    execution_result: Optional[str]    
    error_traceback: Optional[str]     
    retry_count: int                     
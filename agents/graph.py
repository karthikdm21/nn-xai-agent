import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from agents.state      import XAIAgentState
from agents.planner    import run_planner, route_from_planner
from agents.shap_node  import run_shap
from agents.gradcam_node import run_gradcam
from agents.critic     import run_critic, route_from_critic
from agents.narrator   import run_narrator


def build_graph():
    graph = StateGraph(XAIAgentState)

    # add all nodes
    graph.add_node("planner",  run_planner)
    graph.add_node("shap",     run_shap)
    graph.add_node("gradcam",  run_gradcam)
    graph.add_node("critic",   run_critic)
    graph.add_node("narrator", run_narrator)

    # entry point
    graph.set_entry_point("planner")

    # planner decides shap first or gradcam only
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "shap":    "shap",
            "gradcam": "gradcam"
        }
    )

    # shap always goes to gradcam next
    graph.add_edge("shap", "gradcam")

    # gradcam always goes to critic
    graph.add_edge("gradcam", "critic")

    # critic decides to loop or finish
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "planner":  "planner",
            "narrator": "narrator"
        }
    )

    # narrator is the final step
    graph.add_edge("narrator", END)

    return graph.compile()


# build the agent when this module is imported
xai_agent = build_graph()
print("XAI agent graph compiled successfully")
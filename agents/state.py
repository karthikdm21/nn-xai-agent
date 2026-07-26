from typing import TypedDict, List


class XAIAgentState(TypedDict):
    # input
    image_path:      str
    image_tensor:    object

    # model output
    prediction:      str
    confidence:      float
    pred_class_idx:  int

    # tool outputs
    shap_result:     dict
    gradcam_result:  dict

    # agent reasoning
    critique:        str
    contradictions:  List[str]
    next_action:     str

    # final output
    explanation:     str
    confidence_note: str

    # loop control
    loop_count:      int
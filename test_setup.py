import torch
import langgraph
import shap
import mlflow
import fastapi
import streamlit
import captum

print("✅ torch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ langgraph imported")
print("✅ shap imported")
print("✅ mlflow imported")
print("✅ fastapi imported")
print("✅ streamlit imported")
print("✅ captum imported")
print("\n🎉 All good. Day 0 complete.")
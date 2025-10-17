import onnx

model_path = r"models\birefnet.onnx"

model = onnx.load(model_path)
graph = model.graph

print("\n=== INPUTS ===")
for inp in graph.input:
    shape = [d.dim_value if d.dim_value != 0 else "?" for d in inp.type.tensor_type.shape.dim]
    print(f"Name: {inp.name}, Shape: {shape}, Type: {inp.type.tensor_type.elem_type}")

print("\n=== OUTPUTS ===")
for out in graph.output:
    shape = [d.dim_value if d.dim_value != 0 else "?" for d in out.type.tensor_type.shape.dim]
    print(f"Name: {out.name}, Shape: {shape}, Type: {out.type.tensor_type.elem_type}")

print("\n=== DONE ===")
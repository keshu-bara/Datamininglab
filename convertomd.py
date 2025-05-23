import json

# Load the notebook
with open('K-M.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract and format code cells along with outputs
with open('K-M.md', 'w', encoding='utf-8') as f:
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            f.write("```python\n")  # Start of code block
            f.write("".join(cell['source']) + "\n")  # Code content
            f.write("```\n\n")  # End of code block
            
            # Save outputs if available
            if "outputs" in cell:
                for output in cell["outputs"]:
                    if "text" in output:
                        f.write("```output\n")  # Start of output block
                        f.write("".join(output["text"]) + "\n")  # Output content
                        f.write("```\n\n")  # End of output block

print("✅ Code and outputs saved successfully in Markdown!")
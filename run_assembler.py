import os
from token_tamer_core.assembler import ContextAssembler

print("==================================================")
print("🧠 TokenTamer Context Engine Demo (Full Pipeline)")
print("==================================================")

print("\nInitializing Master Assembler...")
# We use the current directory as the codebase
assembler = ContextAssembler(".")

query = "Fix the bug in the token counting logic where the message overhead is calculated incorrectly."
print(f"\nUser Query: \"{query}\"")

print("\nAssembling Context...")
# Let the assembler rank all files based on Semantic, Structural, and Temporal signals.
# We don't specify a target file, forcing it to deduce the target from semantics,
# then pull in dependencies.
ranked_files = assembler.assemble_context(
    query=query,
    top_k=5,
    weights=(0.5, 0.3, 0.2) # Weight semantic intent the highest
)

print("\nTop 5 Most Relevant Files Context Payload:")
print("-" * 50)
for i, (filepath, score) in enumerate(ranked_files, 1):
    filename = os.path.basename(filepath)
    print(f"{i}. [{score:.4f}] {filename}")

print("\nDone.")

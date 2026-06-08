import os
from token_tamer_core.dependency_engine import DependencyEngine
from token_tamer_core.git_engine import GitEngine

print("=" * 50)
print("🧠 TokenTamer Context Engine Demo")
print("=" * 50)

# 1. Test Git Engine
print("\n1. Git Engine (The Temporal Signal)")
print("-" * 50)
git_engine = GitEngine(half_life_days=30)
scores = git_engine.get_recency_scores(".")

if scores:
    print("Files scored by recency (1.0 = newest, 0.0 = oldest):")
    # Sort files by recency score (highest first)
    top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for file_path, score in top_files:
        print(f"  [{score:.4f}] {os.path.basename(file_path)}")
else:
    print("No git history found or empty repository.")

# 2. Test Dependency Engine
print("\n2. Dependency Engine (The Structural Signal)")
print("-" * 50)
# Look for a specific file to analyze
target_file = os.path.abspath("token_tamer/server.py")
if os.path.exists(target_file):
    print(f"Analyzing Target File: {os.path.basename(target_file)}")
    
    engine = DependencyEngine(".")
    deps, dependents = engine.get_dependencies(target_file, depth=1)
    
    print("\nDependencies (What files does it import?):")
    for d in deps:
        print(f"  → {os.path.basename(d)}")
    if not deps:
        print("  (None found in this repo)")
        
    print("\nDependents (What files import it?):")
    for d in dependents:
        print(f"  ← {os.path.basename(d)}")
    if not dependents:
        print("  (None found in this repo)")
else:
    print(f"Could not find {target_file}")

print("\n" + "=" * 50)

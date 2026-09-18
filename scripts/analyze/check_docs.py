import os
import re

docs = [
    "docs/bacr_method.md",
    "docs/experiment_protocol.md",
    "docs/experiment_report.md",
    "docs/research_plan.md",
    "README.md"
]

all_passed = True

for doc in docs:
    if not os.path.exists(doc):
        print(f"[MISSING] {doc}")
        all_passed = False
        continue
        
    with open(doc, "r", encoding="utf-8") as f:
        content = f.read()
        
    dd_count = content.count("$$")
    if dd_count % 2 != 0:
        print(f"[FAIL] {doc}: Odd number of $$ delimiters ({dd_count})")
        all_passed = False
    else:
        # Check alerts
        alerts = re.findall(r'> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', content)
        print(f"[PASS] {doc}: {dd_count // 2} display math blocks, {len(alerts)} GitHub alerts.")

if all_passed:
    print("\nALL DOCUMENTS VALIDATED PERFECTLY!")

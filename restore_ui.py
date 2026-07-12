import json

log_path = "/Users/harihara/.gemini/antigravity-ide/brain/5056f1df-76b5-473a-9337-ff7ba54a6900/.system_generated/logs/transcript_full.jsonl"
code = ""

with open(log_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if data.get("step_index", 9999) > 200:
            break
            
        if data.get("type") == "PLANNER_RESPONSE":
            for tc in data.get("tool_calls", []):
                args = tc.get("args", {})
                target = args.get("TargetFile", "")
                if "app/streamlit_app.py" in target:
                    if tc["name"] == "write_to_file":
                        code = args["CodeContent"]
                    elif tc["name"] == "replace_file_content":
                        # We need to exact string replace
                        tc_target = args.get("TargetContent", "")
                        tc_repl = args.get("ReplacementContent", "")
                        if tc_target in code:
                            code = code.replace(tc_target, tc_repl)

with open("/Users/harihara/Projects/ESG Controversy Detection System/app/streamlit_app.py", "w") as f:
    f.write(code)
print("Restored!")

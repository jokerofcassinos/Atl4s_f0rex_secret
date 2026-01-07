from core.github_uplink import GithubUplink

print("=== INFINITY UPLINK TEST ===")
status = GithubUplink.status()
print(f"Git Status:\n{status}")

if "fatal" not in status.lower():
    print("[SUCCESS] Git connection operational.")
else:
    print("[FAIL] Git not configured or repo missing.")

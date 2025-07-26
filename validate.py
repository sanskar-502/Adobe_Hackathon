# validate.py
import json
import os
from pathlib import Path
from jsonschema import Draft4Validator

SCHEMA_PATH = "sample_dataset/schema/output_schema.json"
OUTPUT_DIR = "sample_dataset/outputs"

def main():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft4Validator(schema)

    passed = 0
    failed = 0

    for json_file in Path(OUTPUT_DIR).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            print(f"FAILED: {json_file.name} failed validation:")
            for error in errors:
                print(f"   ➤ {list(error.path)}: {error.message}")
            failed += 1
        else:
            print(f"PASSED: {json_file.name} passed validation.")
            passed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")

if __name__ == "__main__":
    main()

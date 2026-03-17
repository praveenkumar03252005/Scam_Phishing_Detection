"""
Run this script ONCE locally before uploading to Hugging Face.
It re-saves your model files with pickle protocol 4
which is compatible with Python 3.11.

Usage:
    python resave_model.py
"""

import joblib
import os

FILES = [
    ("scamshield_model.pkl",      "scamshield_model.pkl"),
    ("scamshield_vectorizer.pkl", "scamshield_vectorizer.pkl"),
    # Also handles the old "1" named files if present
    ("scamshield_model1.pkl",     "scamshield_model.pkl"),
    ("scamshield_vectorizer1.pkl","scamshield_vectorizer.pkl"),
    ("scamshield_metadata1.json", None),  # json, no resave needed
]

print("Re-saving model files with pickle protocol 4...\n")

for src, dst in FILES:
    if dst is None:
        # Just rename the metadata file
        if os.path.exists("scamshield_metadata1.json"):
            import shutil
            shutil.copy("scamshield_metadata1.json", "scamshield_metadata.json")
            print("✓ Copied scamshield_metadata1.json → scamshield_metadata.json")
        continue

    if not os.path.exists(src):
        print(f"⏭  '{src}' not found, skipping")
        continue

    print(f"Loading '{src}'...")
    obj = joblib.load(src)
    joblib.dump(obj, dst, protocol=4)
    print(f"✓ Saved  '{dst}' with protocol=4\n")

print("\n✅ Done! Now upload these files to Hugging Face:")
print("   - scamshield_model.pkl")
print("   - scamshield_vectorizer.pkl")
print("   - scamshield_metadata.json")

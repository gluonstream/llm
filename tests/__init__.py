import os

# The suite must never read the real Hugging Face cache. Tests that exercise
# discovery pass an explicit root or ``discovered=`` mapping.
os.environ.setdefault("LMM_NO_CACHE", "1")

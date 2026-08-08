"""
check_setup.py
---------------
Confirms your OpenAI API key works before we build anything on top of it.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into environment variables

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with exactly: setup working"}],
)

print(response.choices[0].message.content)
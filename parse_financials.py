#!/usr/bin/env python3
import os
import argparse
import camelot
import pandas as pd
from openai import OpenAI
from time import sleep


# —— CONFIG —— #
with open("api_key.txt") as f:
    key = f.read().strip()
client = OpenAI(api_key=key)

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """
You are a financial tables classifier.
Given a CSV-like table preview, respond with exactly one of:
- Income Statement
- Balance Sheet
- Cash Flow Statement
- Other
"""

def extract_tables(pdf_path, flavor="lattice"):
    tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
    if not tables:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    return tables

def classify_table(df: pd.DataFrame) -> str:
    preview = df.head(4).to_csv(index=False)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": preview}
        ],
        temperature=0,
        max_tokens=10
    )
    sleep(0.5)
    return resp.choices[0].message.content.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",  required=True, help="PDF file")
    p.add_argument("-o", "--output", required=True, help="Excel .xlsx output")
    args = p.parse_args()

    tables = extract_tables(args.input)
    collected = {
        "Income Statement": [],
        "Balance Sheet":    [],
        "Cash Flow Statement": []
    }

    for idx, table in enumerate(tables, start=1):
        df = table.df.copy()
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        tag = classify_table(df)
        if tag in collected:
            collected[tag].append(df)

    # Write all to one Excel file
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        # Write recognized statements
        for stmt, dfs in collected.items():
            for i, df in enumerate(dfs, start=1):
                sheet = f"{stmt}_{i}"[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)

        # If nothing matched, dump everything under “Other”
        if not any(collected.values()):
            for idx, table in enumerate(tables, start=1):
                df = table.df.copy()
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
                df.to_excel(writer, sheet_name=f"Other_{idx}", index=False)

    print("✅ Done!")

if __name__ == "__main__":
    main()

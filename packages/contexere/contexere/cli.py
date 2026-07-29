"""Contexere CLI tool (`nxt`) for repository indexing and RAG summary generation."""

import os
import sys
import re
import argparse


RAG_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z0-9]+)"
    r"(?P<rag_id>[0-9][0-9][0-9a-zA-Z]{2,3}(?:_[0-9a-zA-Z]+)?)"
    r"__(?P<keyword>.+)"
)


def scan_rag_files(root_dir: str = "."):
    """Scan root_dir for files complying with Contexere RAG naming format.

    Args:
        root_dir: Root directory path to scan.

    Returns:
        List of dictionaries containing parsed file metadata.
    """
    records = []
    ignore_dirs = {".venv", ".git", ".pytest_cache", "__pycache__", "build", "dist", ".agents"}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for filename in sorted(files):
            match = RAG_FILENAME_PATTERN.match(filename)
            if match:
                rel_path = os.path.relpath(os.path.join(root, filename), root_dir)
                prefix = match.group("prefix")
                rag_id = match.group("rag_id")
                keyword = match.group("keyword")

                doc_type = "Document"
                if rel_path.endswith(".ipynb"):
                    doc_type = "Notebook"
                elif rel_path.endswith(".md"):
                    doc_type = "Doc (Minutes)" if "Minutes" in rel_path else "Doc (Markdown)"
                elif rel_path.endswith(".parquet"):
                    doc_type = "Dataset (Parquet)"
                elif rel_path.endswith(".png") or rel_path.endswith(".jpg"):
                    doc_type = "Figure Asset"

                records.append({
                    "prefix": prefix,
                    "rag_id": rag_id,
                    "keyword": keyword,
                    "path": rel_path,
                    "type": doc_type,
                })
    return records


def print_summary(root_dir: str = "."):
    """Print formatted summary table of Contexere RAG indexed documents."""
    records = scan_rag_files(root_dir)

    print("=" * 90)
    print(f"{'Contexere RAG Index Summary':^90}")
    print("=" * 90)

    if not records:
        print("No Contexere RAG compliant files found.")
        print("=" * 90)
        return

    header = f"{'Prefix':<8} {'RAG ID':<10} {'Type':<18} {'Keyword / File Path'}"
    print(header)
    print("-" * 90)

    for rec in records:
        line = f"{rec['prefix']:<8} {rec['rag_id']:<10} {rec['type']:<18} {rec['path']}"
        print(line)

    print("-" * 90)
    print(f"Total Indexed Artifacts: {len(records)}")
    print("=" * 90)


def main():
    """CLI entrypoint for nxt command."""
    parser = argparse.ArgumentParser(
        prog="nxt",
        description="Contexere RAG indexing and context management CLI."
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Generate and display a summary table of Contexere RAG indexed files."
    )
    parser.add_argument(
        "--project", "-p",
        action="store_true",
        help="Display or initialize project RAG index environment."
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="nxt (contexere RAG framework) 0.1.0"
    )

    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.project:
        print("Contexere RAG project environment initialized.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

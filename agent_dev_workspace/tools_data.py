import os
from typing import Dict, Any, List
import pandas as pd
import nltk
from langchain_core.tools import tool

# Ensure NLTK tokenizer resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize text in session.dataframe into unigrams using NLTK word_tokenize.

        Tokenizes lowercased text for all rows in session.dataframe and stores
        the resulting token lists in session.dataframe['unigrams'] and
        session.artifacts['tokens'].

        Returns:
            Dict containing result_id, total_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        # Tokenize lowercased text per document
        tokens_per_doc = session.dataframe["text"].apply(
            lambda t: nltk.word_tokenize(str(t).lower()) if pd.notna(t) and str(t).strip() else []
        ).tolist()

        session.dataframe["unigrams"] = tokens_per_doc
        session.artifacts["tokens"] = tokens_per_doc

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(tokens_per_doc),
            "sample_tokens": tokens_per_doc[:3],
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "newdataset") -> Dict[str, Any]:
        """List files and subdirectories within a given relative directory path.

        Args:
            subdirectory: Relative path to directory (default: 'newdataset').

        Returns:
            Dict containing result_id, subdirectory, files list, and directories list.
        """
        target_path = subdirectory if subdirectory else "."
        if not os.path.exists(target_path):
            return {"error": f"Directory '{target_path}' does not exist."}

        files = []
        directories = []
        for entry in os.listdir(target_path):
            full_p = os.path.join(target_path, entry)
            if os.path.isdir(full_p):
                directories.append(entry)
            else:
                files.append(entry)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "files": sorted(files),
            "directories": sorted(directories),
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspect the first n rows of the working DataFrame in session.

        Args:
            n: Number of initial rows to inspect (default: 5).

        Returns:
            Dict containing result_id, n_rows, columns, and head preview records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        head_df = df.head(n)
        head_records = head_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_rows": len(df),
            "columns": list(df.columns),
            "inspected_rows": len(head_records),
            "head": head_records,
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool]

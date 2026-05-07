from pathlib import Path
import pandas as pd
from langchain_core.documents import Document


def load_excel_structured(file_path: Path):
    docs = []

    try:
        sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in sheets.items():
            df = df.fillna("")

            for i, row in df.iterrows():
                row_text = " | ".join(
                    [f"{col}: {row[col]}" for col in df.columns]
                )

                docs.append(
                    Document(
                        page_content=f"Sheet: {sheet_name}\nRow {i+1}: {row_text}",
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "row": i + 1
                        }
                    )
                )

    except Exception as e:
        print(f"❌ Excel load failed {file_path.name}: {e}")

    return docs
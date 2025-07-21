import pandas as pd
import os
from sqlalchemy import create_engine
from openai import OpenAI
from IPython.display import display, Markdown


class MySQLTableDescriber:
    """
    A class to generate Markdown documentation for a MySQL table,
    including schema overview and LLM-generated column descriptions.
    """
    def __init__(self, schema: str = "example_schema", etl_scripts: dict = None): 
        """
        Initializes the MySQLTableDescriber with connection parameters,
        sets up the Azure OpenAI client, and optionally loads ETL script text.
        etl_scripts: dict (job_name -> job_path), optional.
        """
        
        # MySQL connection parameters
        self.schema = schema
        self.model_name = "gpt-4.1"

        # Initialize the Azure OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_4_1_API_KEY")
        )

        # Load ETL scripts if provided
        self.etl_scripts_content = self._load_etl_scripts(etl_scripts) if etl_scripts is not None else None


    def _load_etl_scripts(self, etl_scripts: dict) -> dict:
        """
        Loads ETL script files based on a dictionary {job_name: file_path}.
        Returns a dictionary {job_name: code}.
        """
        result = {}
        for job, path in etl_scripts.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result[job] = f.read()
            except Exception as e:
                result[job] = f"ERROR: Could not read file {path}: {e}"
        return result


    def MySQL_connect(self, schema: str) -> str:
        """
        Builds a MySQL connection string for SQLAlchemy.

        Reads connection parameters from environment variables:
        - MYSQL_USER
        - MYSQL_PASSWORD
        - MYSQL_HOST
        - MYSQL_PORT

        """
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{schema}?charset=utf8mb4"


    def run_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.
        This method connects to the MySQL database using SQLAlchemy.
        """
        engine = create_engine(self.MySQL_connect(self.schema))
        conn = engine.connect()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df


    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """
        Retrieve schema information for the table (columns, types, nullability, default, primary key).
        """
        query = f"SHOW COLUMNS FROM {self.schema}.{table_name};"
        schema_df = self.run_sql(query)
        return schema_df


    def get_sample_rows(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Fetch a few sample rows from the table for context (default limit).
        Now: fetch 100 rows from SQL, then select limit random rows using pandas.
        """
        query = f"SELECT * FROM {self.schema}.{table_name} LIMIT 100;"
        df = self.run_sql(query)
        if len(df) > limit:
            df = df.sample(limit, random_state=42)
        return df


    def _gather_sample_values(self, schema_df: pd.DataFrame, table_name: str) -> dict:
        """
        For categorical columns, it collects all unique values.
        For transactional columns, it collects a sample of values.
        Improved heuristic:
        - categorical: num_unique < 100 and sum of top 5 value frequencies >= 30% of all rows
        - transactional: otherwise
        """
        sample_values = {}
        for _, row in schema_df.iterrows():
            col = row["Field"]
            col_type = str(row["Type"]).lower()
            if "char" in col_type or "text" in col_type:
                query = f"SELECT {col} FROM {self.schema}.{table_name} WHERE {col} IS NOT NULL"
                df_col = self.run_sql(query)
                unique_vals = df_col[col].dropna().unique().tolist()
                num_unique = len(unique_vals)
                total_count = len(df_col)
                value_counts = df_col[col].value_counts()
                top5_sum = value_counts.iloc[:5].sum() if len(value_counts) >= 5 else value_counts.sum()
                top5_ratio = top5_sum / total_count if total_count > 0 else 0
                if num_unique < 100 and top5_ratio >= 0.3:
                    sample_values[col] = {
                        "type": "categorical",
                        "values": [str(v) for v in unique_vals]
                    }
                else:
                    sample_values[col] = {
                        "type": "transactional",
                        "values": [str(v) for v in unique_vals[:3]]
                    }
        return sample_values


    def describe_table(self, table_name: str) -> str:
        """
        Generate a markdown documentation string for the given table using MySQL data
        and LLM for descriptions.
        """
        # Retrieve schema and sample data
        schema_df = self.get_table_schema(table_name)
        sample_df = self.get_sample_rows(table_name, limit=5)
        sample_values = self._gather_sample_values(schema_df, table_name)

        # Build Markdown table of schema
        md = f"### Table: {table_name}\n"
        md += "| Field | Type | Null | Default | Key |\n"
        md += "|-------|------|------|---------|-----|\n"
        for _, row in schema_df.iterrows():
            field = row["Field"]
            dtype = row["Type"]
            null = row["Null"]
            default = row["Default"]
            key = row["Key"]
            md += f"| {field} | {dtype} | {null} | {default} | {key} |\n"

        # Build Markdown table of sample rows
        if not sample_df.empty:
            md += "\n#### Sample rows:\n"
            md += "| " + " | ".join([f"{col}" for col in sample_df.columns]) + " |\n"
            md += "| " + " | ".join(["---"] * len(sample_df.columns)) + " |\n"
            for _, row in sample_df.iterrows():
                md += "| " + " | ".join([str(row[col]) for col in sample_df.columns]) + " |\n"

        # Prepare column info for prompt
        column_info = []
        for _, row in schema_df.iterrows():
            col_name = row["Field"]
            col_type = row["Type"]
            if col_name in sample_values:
                info = sample_values[col_name]
                if info["type"] == "categorical":
                    example_text = f" Categorical values: {info['values']}."
                else:
                    example_text = f" Sample transactional values: {info['values']}."
            else:
                example_text = ""
            column_info.append(f"**{col_name}** (Type: {col_type}).{example_text}")
        info_str = "\n".join(f"- {ci}" for ci in column_info)

        system_prompt = (
            "You are a data expert tasked with documenting a database table. "
            "Provide clear, concise, and professional descriptions for each column."
        )
        # Add ETL code sections if present
        etl_section = ""
        if self.etl_scripts_content:
            for job, code in self.etl_scripts_content.items():
                etl_section += f"\nETL script for job: {job}\npython\n{code}\n\n"

        user_prompt = (
            f"Table schema:\n{md}\n\n"
            f"Columns in table {table_name}:\n{info_str}\n\n"
            f"{etl_section}"
            "Please follow these instructions:\n"
            "1) Specify the data type of the column.\n"
            "2) Begin each column description with 'Pochodzenie: ...', specifying the data origin (e.g. Excel, MySQL table, API, ERP or other system) based on the ETL script if possible.\n"
            "3) After 'Pochodzenie: ...' insert two newlines before the rest of the description.\n"
            "4) If the origin cannot be determined, use 'Pochodzenie: Nieznane'.\n"
            "5) For each column, write a description explaining its purpose or contents.\n"
            "6) If the column is derived from any calculations, describe these calculations.\n"
            "7) All descriptions must be in Polish.\n"
            "8) Return the result as a markdown table (without blocks) in the following format:\n\n"
            "| Kolumna | Opis |\n"
            "|---------|------|\n"
            "| `ExampleColumn` | **Typ:** _Data type_ (one newline) **Pochodzenie:** _Excel(workbook name)/MySQL (schema.table_name) etc._ (two newline) **Opis:** _Description of the column_ |"
        )

        # Print the prompts for debugging
        display(Markdown(f"System Prompt: {system_prompt}\n"))
        display(Markdown(f"User Prompt: {user_prompt}"))

        # Call the Azure OpenAI client
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        descriptions = response.choices[0].message.content.strip()
        
        print(f"Input tokens used: {response.usage.prompt_tokens}")
        print(f"Output tokens used: {response.usage.completion_tokens}")
        print(f"Tokens used: {response.usage.total_tokens}")

        return descriptions

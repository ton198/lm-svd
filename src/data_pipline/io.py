from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_atomic(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        pq.write_table(table, temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

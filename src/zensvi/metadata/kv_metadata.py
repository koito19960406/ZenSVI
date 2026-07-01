import polars as pl

from zensvi.metadata.base_metadata import BaseMetadata


class KVMetadata(BaseMetadata):
    """Compute metadata for a KartaView (KV) dataset.

    Normalizes the columns of a KartaView ``kv_pids.csv`` (produced by ``KVDownloader``)
    to the schema expected by :class:`BaseMetadata`:

    - ``shotDate`` (``"%Y-%m-%d %H:%M:%S%.f"``, UTC) -> ``captured_at`` (ms since Unix epoch)
    - ``heading`` -> ``compass_angle``
    - ``sequenceId`` -> ``sequence_id``
    - ``userId`` -> ``creator_id``
    - ``orgCode`` -> ``organization_id``
    - ``id``, ``lat``, ``lon``, ``is_pano`` are used as-is.

    Args:
        path_input (Union[str, Path]): path to the input CSV file (e.g. ``kv_pids.csv``).
        log_path (Union[str, Path], optional): Path to the log file. Defaults to None.
    """

    _COLUMN_RENAMES = {
        "heading": "compass_angle",
        "sequenceId": "sequence_id",
        "userId": "creator_id",
        "orgCode": "organization_id",
    }

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map KartaView columns to the normalized metadata schema."""
        # shotDate (naive UTC timestamp string) -> captured_at (ms since Unix epoch)
        df = df.with_columns(
            pl.col("shotDate")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
            .dt.replace_time_zone("UTC")
            .dt.epoch(time_unit="ms")
            .alias("captured_at")
        )
        renames = {src: dst for src, dst in self._COLUMN_RENAMES.items() if src in df.columns}
        df = df.rename(renames)
        if "is_pano" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("is_pano"))
        return df

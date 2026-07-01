from zensvi.metadata.base_metadata import BaseMetadata


class MLYMetadata(BaseMetadata):
    """Compute metadata for a Mapillary (MLY) dataset.

    The input CSV (e.g. ``mly_pids.csv``) already uses the schema expected by
    :class:`BaseMetadata` — ``"id", "lat", "lon", "captured_at", "compass_angle",
    "creator_id", "sequence_id", "organization_id", "is_pano"`` — so no column
    normalization is needed.

    Args:
        path_input (Union[str, Path]): path to the input CSV file (e.g. ``mly_pids.csv``).
        log_path (Union[str, Path], optional): Path to the log file. Defaults to None.
    """

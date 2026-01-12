from logging import Logger
from pathlib import Path
from typing import List

from ..data import AnnotationTable, WorldlineTable
from ._utilities import default_args

@default_args("_this, _now, *, _ACTIVE")
def delete_active_track(
    dataset: Path,
    annotations: AnnotationTable,
    worldlines: WorldlineTable,
    window_state,
    arg: str,
    logger: Logger
) -> List[dict]:
    """
    Delete the currently selected track and all its annotations.
    """

    selected_worldline_id = window_state['selected_worldline']

    if selected_worldline_id is None:
        logger.warning("No track selected to delete.")
        return []

    # Get integer ID
    try:
        wl_id = int(selected_worldline_id)
    except (ValueError, TypeError):
        logger.error(f"Invalid worldline ID: {selected_worldline_id}")
        return []

    # 1. Delete associated annotations (Cascading Delete Logic)
    if hasattr(annotations, 'df'):
        try:
            # Find all annotations with this worldline_id
            # Using dataframe filtering assumes pandas backend which seems true based on data/io.py
            ids_to_delete = annotations.df[annotations.df['worldline_id'] == wl_id]['id'].tolist()
            if ids_to_delete:
                annotations.delete_ids(ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} annotations for worldline {wl_id}")
        except Exception as e:
            logger.error(f"Error cleaning up annotations for worldline {wl_id}: {e}")

    # 2. Delete the worldline itself
    worldlines.delete(wl_id)
    logger.info(f"Deleted worldline {wl_id}")

    # 3. Return actions to update frontend state
    return [
        {
            "type": "worldlines/delete_worldline_local",
            "payload": int(wl_id)
        },
        {
            "type": "annotation_window/set_selected_worldline",
            "payload": None
        },
        {
            "type": "annotations/get_annotations"  # Refresh annotations just in case
        }
    ]

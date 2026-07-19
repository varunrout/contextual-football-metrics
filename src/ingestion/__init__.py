"""
src.ingestion
=============
Data ingestion and internal schema for the contextual football metric suite.

Public surface:
  schema          — internal dataclass types and enumerations
  statsbomb_loader — StatsBomb Open Data fetch + cache
  provider_mapper  — raw StatsBomb → internal schema conversion
  possession_builder — possession reconstruction from event stream
  data_qa          — data integrity and coverage QA
"""

# Sub-modules are imported on demand; import them explicitly where needed.
# e.g.:  from src.ingestion.schema import ShotEvent
#        from src.ingestion.statsbomb_loader import load_events

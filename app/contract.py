"""API/index contract constants shared by stores — kept dependency-free so
pg_index does not need app.rebuild (which lazily imports it back)."""

CONTRACT_VERSION = "v1"

#!/bin/bash
# Quick ingestion script — run after containers are up
set -e
echo "Triggering document ingestion..."
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_dir": "data/documents", "force_reload": false}' | python3 -m json.tool
echo ""
echo "Checking health..."
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool

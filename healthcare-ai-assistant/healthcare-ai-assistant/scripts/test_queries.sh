#!/bin/bash
# Run sample queries against the running API to verify correctness
set -e
BASE="http://localhost:8000/api/v1"

echo "====== HEALTHCARE AI ASSISTANT — QUERY TESTS ======"
echo ""

ask() {
  local label="$1"
  local query="$2"
  echo "--- $label ---"
  curl -s -X POST "$BASE/ask" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"include_sources\": true}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Q:', d.get('query',''))
print('A:', d.get('answer','')[:200], '...')
print('Confidence:', d.get('confidence',''), '|', d.get('confidence_score',0))
print('Type:', d.get('query_type',''))
print('Sources:', [s['document_name'] for s in d.get('sources',[])])
print()
"
}

ask "HIPAA Rights" "What are my rights under HIPAA?"
ask "Telehealth Hours" "What are the telehealth visit hours?"
ask "Appointment Booking" "I want to book an appointment"
ask "Medication Refill" "How do I request a prescription refill?"
ask "Insurance" "Which insurance plans do you accept?"
ask "Discharge - Emergency Signs" "What emergency signs should I watch for after discharge?"
ask "Out of Scope" "What is the recipe for chocolate cake?"
ask "Scheduling - slots" "Show me available slots this week"

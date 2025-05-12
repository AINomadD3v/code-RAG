#!/usr/bin/env bash
set -e

echo "📦 [1/5] Pulling Postgres + pgvector Docker image..."
docker compose -f docker-compose.yml pull || true

echo "🐘 [2/5] Starting Postgres container..."
docker compose -f docker-compose.yml up -d

echo "⌛ [3/5] Waiting for Postgres to be ready..."
sleep 5
until docker exec cocoindex_postgres pg_isready -U cocoindex > /dev/null 2>&1; do
  sleep 1
done

echo "🔧 [4/5] Verifying DB connection..."
docker exec -e PGPASSWORD=cocoindex cocoindex_postgres \
  psql -U cocoindex -d cocoindex -c "\\l"

echo ""
echo "✅ [5/5] Postgres is ready."
echo "🧠 Connection string (for CocoIndex or LangChain):"
echo "    postgres://cocoindex:cocoindex@localhost:5432/cocoindex"
echo ""
echo "🚀 You can now run your pipeline or start:"
echo "    cocoinsight serve"


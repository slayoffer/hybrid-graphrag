#!/bin/bash

echo "🚀 Setting up LightRAG Neo4j Production System"
echo "=============================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your credentials before proceeding!"
    exit 1
fi

# Start Neo4j
echo "🗄️  Starting Neo4j database..."
docker-compose up -d

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
sleep 10

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create indexes
echo "🔍 Creating database indexes..."
python create_indexes.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run 'python ingest_documents.py' to ingest documents"
echo "2. Run 'python query_rag.py' to query the system"
echo ""
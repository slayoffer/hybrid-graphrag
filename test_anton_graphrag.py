#!/usr/bin/env python3
"""Test GraphRAG with all three modes on Anton Evseev's About Me document."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys
from src.rag_system import RAGSystem
import time

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# 10 Test questions about Anton Evseev
TEST_QUESTIONS = [
    # Simple factual questions
    {
        "id": 1,
        "question": "What is Anton Evseev's current position at Xsolla?",
        "expected_answer": "DevSecOps Engineer",
        "difficulty": "simple"
    },
    {
        "id": 2,
        "question": "When did Anton start working at Xsolla?",
        "expected_answer": "December 1st, 2024",
        "difficulty": "simple"
    },
    
    # Background questions
    {
        "id": 3,
        "question": "What is Anton's educational background?",
        "expected_answer": "Bachelor's degree in Oriental Studies, South Korean Relations & Economy from Far Eastern State University",
        "difficulty": "medium"
    },
    {
        "id": 4,
        "question": "What companies has Anton founded or co-founded?",
        "expected_answer": "Co-founder & CTO at MEDUNION Korea, Founder & CTO at LIFEKOREA",
        "difficulty": "medium"
    },
    
    # Current responsibilities
    {
        "id": 5,
        "question": "What compliance standards does Anton ensure systems meet at Xsolla?",
        "expected_answer": "PCI DSS, GDPR, and other financial security requirements",
        "difficulty": "complex"
    },
    {
        "id": 6,
        "question": "What languages can Anton speak?",
        "expected_answer": "Russian, English and Korean",
        "difficulty": "simple"
    },
    
    # Personal information
    {
        "id": 7,
        "question": "What is Anton's Myers-Briggs personality type?",
        "expected_answer": "INFJ-A Advocate",
        "difficulty": "medium"
    },
    {
        "id": 8,
        "question": "What courses did Anton recently complete?",
        "expected_answer": "Automate Your Deployments on Kubernetes with GitHub Actions and Argo GitOps, Gitlab DevSecOps, and DevSecOps Bootcamp",
        "difficulty": "complex"
    },
    
    # Curiosities and interests
    {
        "id": 9,
        "question": "What unusual server does Anton run and how many bots are on it?",
        "expected_answer": "WoW Classic server with 5000 bots",
        "difficulty": "complex"
    },
    {
        "id": 10,
        "question": "Where does Anton currently live and where would he like to live?",
        "expected_answer": "Currently lives in Vladivostok, Russia, wants to move to a rural area with his own house",
        "difficulty": "complex"
    }
]


async def evaluate_answer(answer, expected):
    """Evaluate answer accuracy."""
    
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    # Check for key terms
    expected_terms = set(expected_lower.replace(",", "").replace(".", "").replace("and", "").split())
    answer_terms = set(answer_lower.replace(",", "").replace(".", "").replace("and", "").split())
    
    # Calculate overlap
    overlap = len(expected_terms & answer_terms) / len(expected_terms) if expected_terms else 0
    
    # Bonus for important terms specific to Anton's profile
    important_matches = {
        "devsecops": 0.2,
        "december": 0.15,
        "2024": 0.15,
        "oriental studies": 0.2,
        "korean": 0.15,
        "medunion": 0.15,
        "lifekorea": 0.15,
        "pci dss": 0.2,
        "gdpr": 0.2,
        "infj": 0.25,
        "advocate": 0.15,
        "wow classic": 0.2,
        "5000": 0.15,
        "vladivostok": 0.2,
        "kubernetes": 0.15,
        "gitlab": 0.15
    }
    
    bonus = 0
    for term, weight in important_matches.items():
        if term in expected_lower and term in answer_lower:
            bonus += weight
    
    return min(overlap + bonus, 1.0)


async def test_mode(rag, mode, questions):
    """Test a specific search mode."""
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing {mode.upper()} mode")
    logger.info(f"{'=' * 60}")
    
    results = []
    
    for q_data in questions:
        logger.info(f"\nQuestion {q_data['id']} ({q_data['difficulty']}): {q_data['question']}")
        logger.info(f"Expected: {q_data['expected_answer']}")
        
        # Measure time
        start_time = time.time()
        
        # Query using specified mode
        answer = await rag.query(q_data["question"], search_mode=mode)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Generated Answer: {answer[:200]}...")
        
        # Evaluate
        score = await evaluate_answer(answer, q_data["expected_answer"])
        
        results.append({
            "id": q_data["id"],
            "question": q_data["question"],
            "difficulty": q_data["difficulty"],
            "expected": q_data["expected_answer"],
            "generated": answer,
            "score": score,
            "latency_ms": latency_ms
        })
        
        logger.info(f"Score: {score:.2%} | Latency: {latency_ms:.0f}ms")
    
    return results


async def main():
    """Test RAG system with all three modes."""
    
    logger.info("=" * 80)
    logger.info("Testing GraphRAG System with Anton Evseev's Profile")
    logger.info("=" * 80)
    
    # Initialize RAG system with correct workspace
    rag = RAGSystem(workspace="anton_profile")
    
    # Check database stats
    stats = await rag.get_statistics()
    logger.info(f"\n📊 Database Stats:")
    logger.info(f"  Entities: {stats['entities']}")
    logger.info(f"  Chunks: {stats['chunks']}")
    logger.info(f"  Relationships: {stats['relationships']}")
    
    if stats['chunks'] == 0:
        logger.error("❌ No data found! Please run ingestion first.")
        await rag.close()
        return
    
    # Check KNN graph stats
    async with rag.driver.session() as session:
        knn_stats = await rag.knn_graph.get_graph_stats(session, "anton_profile")
        logger.info(f"\n🔗 KNN Graph Stats:")
        logger.info(f"  Total relationships: {knn_stats['total_relationships']}")
        logger.info(f"  Avg similarity score: {knn_stats['avg_similarity_score']:.3f}")
        logger.info(f"  Avg neighbors per chunk: {knn_stats['avg_neighbors_per_chunk']:.1f}")
    
    all_results = {}
    
    # Test all modes
    for mode in ["vector", "graph", "hybrid"]:
        mode_results = await test_mode(rag, mode, TEST_QUESTIONS)
        all_results[mode] = mode_results
    
    # Generate summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for mode in ["vector", "graph", "hybrid"]:
        results = all_results[mode]
        
        by_difficulty = {}
        for r in results:
            diff = r["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(r["score"])
        
        logger.info(f"\n{mode.upper()} MODE:")
        
        for diff in ["simple", "medium", "complex"]:
            if diff in by_difficulty:
                scores = by_difficulty[diff]
                avg_score = sum(scores) / len(scores) if scores else 0
                logger.info(f"  {diff.capitalize()}: {len(scores)} questions, avg score: {avg_score:.2%}")
        
        overall_score = sum(r["score"] for r in results) / len(results) if results else 0
        avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        logger.info(f"  OVERALL: {overall_score:.2%} accuracy, {avg_latency:.0f}ms avg latency")
    
    # Save results
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"anton_graphrag_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "document": "About Me - Anton Evseev",
            "stats": {
                "chunks": stats['chunks'],
                "knn_relationships": knn_stats['total_relationships'],
                "avg_neighbors": knn_stats['avg_neighbors_per_chunk']
            },
            "results": all_results,
            "summary": {
                mode: {
                    "overall_score": sum(r["score"] for r in results) / len(results),
                    "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results)
                }
                for mode, results in all_results.items()
            }
        }, f, indent=2)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    # Find best performing mode
    best_mode = max(all_results.keys(), key=lambda m: sum(r["score"] for r in all_results[m]))
    best_score = sum(r["score"] for r in all_results[best_mode]) / len(all_results[best_mode])
    logger.info(f"\n🏆 Best performing mode: {best_mode.upper()} with {best_score:.2%} accuracy")
    
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
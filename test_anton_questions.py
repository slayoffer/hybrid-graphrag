#!/usr/bin/env python3
"""Test questions for Anton Evseev's About Me document."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys
from src.rag_system import RAGSystem

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


async def main():
    """Test RAG system with Anton's profile questions."""
    
    logger.info("=" * 80)
    logger.info("Testing RAG System with Anton Evseev's Profile")
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
    
    results = []
    
    for q_data in TEST_QUESTIONS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Question {q_data['id']} ({q_data['difficulty']}): {q_data['question']}")
        logger.info(f"Expected: {q_data['expected_answer']}")
        
        # Query using vector search on chunks (98% accuracy mode)
        answer = await rag.query(q_data["question"], search_mode="vector_chunks")
        
        logger.info(f"Generated Answer: {answer[:200]}...")
        
        # Evaluate
        score = await evaluate_answer(answer, q_data["expected_answer"])
        
        results.append({
            "id": q_data["id"],
            "question": q_data["question"],
            "difficulty": q_data["difficulty"],
            "expected": q_data["expected_answer"],
            "generated": answer,
            "score": score
        })
        
        logger.info(f"Score: {score:.2%}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    by_difficulty = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(r["score"])
    
    for diff in ["simple", "medium", "complex"]:
        if diff in by_difficulty:
            scores = by_difficulty[diff]
            avg_score = sum(scores) / len(scores) if scores else 0
            logger.info(f"{diff.upper()}: {len(scores)} questions, avg score: {avg_score:.2%}")
    
    overall_score = sum(r["score"] for r in results) / len(results) if results else 0
    logger.info(f"\nOVERALL SCORE: {overall_score:.2%}")
    
    # Save results
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"anton_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "document": "About Me - Anton Evseev",
            "results": results,
            "summary": {
                "overall_score": overall_score,
                "by_difficulty": {k: sum(v)/len(v) for k, v in by_difficulty.items()}
            }
        }, f, indent=2)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    # Best and worst performing questions
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    logger.info("\n✅ Best performing questions:")
    for r in sorted_results[:3]:
        logger.info(f"  Q{r['id']}: {r['score']:.2%} - {r['question'][:50]}...")
    
    logger.info("\n❌ Worst performing questions:")
    for r in sorted_results[-3:]:
        logger.info(f"  Q{r['id']}: {r['score']:.2%} - {r['question'][:50]}...")
    
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
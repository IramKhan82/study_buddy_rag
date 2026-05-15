"""
test_rag.py
───────────
Basic automated tests for the RAG pipeline.
Run:  python test_rag.py
Requires:  GEMINI_API_KEY env var  +  a sample PDF named sample.pdf
"""

import os
import sys
import textwrap
import tempfile

# ── Minimal synthetic PDF creation (no external PDF needed for unit tests) ──
def _make_synthetic_pdf(path: str) -> None:
    """Create a tiny single-page PDF with known text using PyMuPDF."""
    import fitz
    doc  = fitz.open()
    page = doc.new_page()
    content = textwrap.dedent("""
        Chapter 1: Introduction to Photosynthesis
        Photosynthesis is the process by which green plants convert sunlight into food.
        The chemical equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
        Chlorophyll is the pigment responsible for absorbing light energy.
        Plants use glucose produced from photosynthesis for growth and energy.

        Chapter 2: Cellular Respiration
        Cellular respiration breaks down glucose to release energy (ATP).
        It occurs in the mitochondria of cells.
        The equation is: C6H12O6 + 6O2 → 6CO2 + 6H2O + ATP.
    """).strip()
    page.insert_text((50, 50), content, fontsize=11)
    doc.save(path)
    doc.close()


def run_tests():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set — skipping live API tests.\n"
              "   Set it and re-run for full tests.")
        sys.exit(0)

    from rag_engine import RAGEngine

    print("=" * 60)
    print("Study Buddy RAG — Test Suite")
    print("=" * 60)
    passed = failed = 0

    # ── Create a temp PDF ────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    _make_synthetic_pdf(pdf_path)

    engine = RAGEngine(gemini_api_key=api_key, chunk_size=400, chunk_overlap=50)

    # Test 1: Ingestion
    try:
        n = engine.ingest(pdf_path)
        assert n > 0, "Expected at least 1 chunk"
        print(f"[PASS] Ingestion — {n} chunks indexed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Ingestion — {e}")
        failed += 1

    # Test 2: In-scope question
    try:
        r = engine.query("What is photosynthesis?", top_k=3)
        assert not r["out_of_scope"], "Expected in-scope answer"
        assert len(r["chunks"]) > 0
        print(f"[PASS] In-scope Q — answer: {r['answer'][:80]}…")
        passed += 1
    except Exception as e:
        print(f"[FAIL] In-scope Q — {e}")
        failed += 1

    # Test 3: Out-of-scope question
    try:
        r = engine.query("What is the capital of France?", top_k=3)
        assert r["out_of_scope"], "Expected out-of-scope detection"
        print(f"[PASS] Out-of-scope Q — correctly refused")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Out-of-scope Q — {e}")
        failed += 1

    # Test 4: Chunk retrieval count
    try:
        r = engine.query("Explain cellular respiration.", top_k=2)
        assert len(r["chunks"]) <= 2
        print(f"[PASS] Top-K retrieval — {len(r['chunks'])} chunks returned")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Top-K retrieval — {e}")
        failed += 1

    os.unlink(pdf_path)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run_tests()

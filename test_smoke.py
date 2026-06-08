"""Smoke test for TokenTamer core components."""

from token_tamer.skeletonizer import Skeletonizer
from token_tamer.context_analyzer import ContextAnalyzer
from token_tamer.token_counter import TokenCounter

# ── Test 1: Skeletonizer ──
print("=" * 60)
print("TEST 1: AST Skeletonizer")
print("=" * 60)

code = '''import os

class PaymentProcessor:
    """Process payments for customers."""

    base_rate: float = 0.05

    def calculate_tax(self, amount: float, region: str) -> float:
        rate = self.get_base_rate(region)
        adjustments = self.fetch_adjustments(region, amount)
        if amount > 10000:
            rate *= 1.05
        subtotal = amount * rate
        for adj in adjustments:
            subtotal += adj.value
        return round(subtotal, 2)

    def process(self, payment_id: str, amount: float) -> dict:
        tax = self.calculate_tax(amount, "US")
        total = amount + tax
        record = {
            "id": payment_id,
            "amount": amount,
            "tax": tax,
            "total": total,
        }
        self.save_to_db(record)
        self.send_notification(payment_id)
        return record

    def save_to_db(self, record: dict) -> None:
        db = get_connection()
        db.insert("payments", record)
        db.commit()


def helper_function(x: int, y: int) -> int:
    result = x * y + 42
    for i in range(result):
        if i % 7 == 0:
            result += i
    return result
'''

s = Skeletonizer(keep_class_attrs=True)
result = s.skeletonize(code)
print("\n--- SKELETON OUTPUT ---")
print(result.skeleton)
print(f"\nOriginal lines: {result.original_lines}")
print(f"Skeleton lines: {result.skeleton_lines}")
print(f"Lines saved:    {result.lines_saved}")
print(f"Compression:    {result.compression_ratio:.1%}")
print(f"Was compressed: {result.was_compressed}")

# ── Test 2: Context Analyzer ──
print("\n" + "=" * 60)
print("TEST 2: Context Analyzer")
print("=" * 60)

analyzer = ContextAnalyzer(s)
messages = [
    {
        "role": "user",
        "content": (
            "Fix the bug in payment.py where tax calculation is wrong.\n\n"
            "```python\n"
            "# File: payment.py\n"
            "def calculate_tax(amount, region):\n"
            "    rate = get_rate(region)\n"
            "    return amount * rate\n"
            "```\n\n"
            "```python\n"
            "# File: database.py\n"
            "def get_connection():\n"
            "    conn = sqlite3.connect('app.db')\n"
            "    conn.row_factory = sqlite3.Row\n"
            "    return conn\n\n"
            "def execute_query(conn, query, params=None):\n"
            "    cursor = conn.cursor()\n"
            "    if params:\n"
            "        cursor.execute(query, params)\n"
            "    else:\n"
            "        cursor.execute(query)\n"
            "    return cursor.fetchall()\n"
            "```\n"
        )
    }
]

active_files = analyzer.extract_active_files(messages)
print(f"\nActive files detected: {active_files}")

compressed_msgs, analysis = analyzer.analyze_and_compress(messages)
print(f"Total code blocks: {analysis.total_blocks}")
print(f"Skeletonized blocks: {analysis.skeletonized_blocks}")

for block in analysis.code_blocks:
    status = "INTACT" if block.is_active else "SKELETONIZED"
    print(f"  [{status}] {block.filename or 'unknown'}")

# ── Test 3: Token Counter ──
print("\n" + "=" * 60)
print("TEST 3: Token Counter")
print("=" * 60)

counter = TokenCounter()

original_tokens = counter.count_messages(messages)
compressed_tokens = counter.count_messages(compressed_msgs)
saved = original_tokens - compressed_tokens

print(f"\nOriginal tokens:   {original_tokens}")
print(f"Compressed tokens: {compressed_tokens}")
print(f"Tokens saved:      {saved}")
if original_tokens > 0:
    print(f"Reduction:         {saved / original_tokens:.1%}")

print("\n✅ All smoke tests passed!")

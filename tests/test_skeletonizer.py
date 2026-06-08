"""Tests for the multi-language skeletonizer."""

import pytest
from token_tamer.skeletonizer import Skeletonizer, C_STYLE_LANGUAGES


PYTHON_CODE = """import os

class PaymentProcessor:
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
        record = {"id": payment_id, "amount": amount, "tax": tax, "total": total}
        self.save_to_db(record)
        self.send_notification(payment_id)
        return record

def helper_function(x: int, y: int) -> int:
    result = x * y + 42
    for i in range(result):
        if i % 7 == 0:
            result += i
    return result
"""

JS_CODE = """import { getRate } from './rates';

export class PaymentProcessor {
    constructor(baseRate) {
        this.baseRate = baseRate;
    }

    calculateTax(amount, region) {
        const rate = getRate(region);
        let subtotal = amount * rate;
        if (subtotal > 10000) {
            subtotal *= 1.05;
        }
        return Math.round(subtotal, 2);
    }

    async process(paymentId, amount) {
        const tax = this.calculateTax(amount, 'US');
        const total = amount + tax;
        await this.saveToDb({ paymentId, amount, tax, total });
        return { paymentId, total };
    }
}

function helper(x, y) {
    let result = x * y + 42;
    for (let i = 0; i < result; i++) {
        if (i % 7 === 0) result += i;
    }
    return result;
}
"""

GO_CODE = """package main

import "fmt"

func calculateTax(amount float64, region string) float64 {
    rate := getBaseRate(region)
    adjustments := fetchAdjustments(region, amount)
    if amount > 10000 {
        rate *= 1.05
    }
    subtotal := amount * rate
    for _, adj := range adjustments {
        subtotal += adj.Value
    }
    return subtotal
}

func processPayment(paymentID string, amount float64) map[string]interface{} {
    tax := calculateTax(amount, "US")
    total := amount + tax
    record := map[string]interface{}{
        "id": paymentID,
        "amount": amount,
        "tax": tax,
        "total": total,
    }
    saveToDB(record)
    return record
}
"""


class TestPythonSkeletonizer:
    def test_compresses_functions(self, skeletonizer: Skeletonizer):
        result = skeletonizer.skeletonize(PYTHON_CODE, language="python")
        assert result.was_compressed is True
        assert result.compression_ratio > 0.3
        assert "def calculate_tax(self, amount: float, region: str) -> float:" in result.skeleton
        assert "def process(self, payment_id: str, amount: float) -> dict:" in result.skeleton
        assert "..." in result.skeleton

    def test_preserves_class_attrs(self, skeletonizer: Skeletonizer):
        result = skeletonizer.skeletonize(PYTHON_CODE, language="python")
        assert "base_rate: float = 0.05" in result.skeleton

    def test_fallback_on_syntax_error(self, skeletonizer: Skeletonizer):
        bad_code = "def foo(:\n"
        result = skeletonizer.skeletonize(bad_code, language="python")
        assert result.was_compressed is False
        assert result.skeleton == bad_code


class TestCStyleSkeletonizer:
    def test_javascript_compression(self, skeletonizer: Skeletonizer):
        result = skeletonizer.skeletonize(JS_CODE, language="javascript")
        assert result.was_compressed is True
        assert "calculateTax(amount, region)" in result.skeleton
        assert "..." in result.skeleton
        assert "import { getRate }" in result.skeleton

    def test_go_compression(self, skeletonizer: Skeletonizer):
        result = skeletonizer.skeletonize(GO_CODE, language="go")
        assert result.was_compressed is True
        assert "func calculateTax(amount float64, region string)" in result.skeleton
        assert "..." in result.skeleton
        assert 'import "fmt"' in result.skeleton

    @pytest.mark.parametrize("lang", ["typescript", "rust", "java", "c", "cpp"])
    def test_all_cstyle_languages_return_something(self, skeletonizer: Skeletonizer, lang: str):
        """Every C-style language should at least attempt compression."""
        code = "function foo() {\n    return 42;\n}\n"
        result = skeletonizer.skeletonize(code, language=lang)
        assert result.was_compressed is True
        assert "..." in result.skeleton


class TestUnknownLanguage:
    def test_unknown_language_passthrough(self, skeletonizer: Skeletonizer):
        code = "some random text\nmore text\n"
        result = skeletonizer.skeletonize(code, language="brainfuck")
        assert result.was_compressed is False
        assert result.skeleton == code

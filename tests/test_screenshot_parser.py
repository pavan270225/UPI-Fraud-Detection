"""
Tests for screenshot parser module.

Tests the ParsedTransaction dataclass, ScreenshotParser class, and parse_screenshot
convenience function. Uses mocked OCR to avoid requiring actual image files.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Skip if OCR not available
try:
    from src.screenshot_parser import (
        ScreenshotParser,
        ParsedTransaction,
        ScreenshotParserError,
        parse_screenshot,
        OCR_AVAILABLE
    )
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False

if HAS_PARSER:
    from src.screenshot_parser import OCR_AVAILABLE

pytestmark = pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR dependencies not installed")


class TestParsedTransaction:
    """Tests for ParsedTransaction dataclass."""

    def test_to_features_basic(self):
        txn = ParsedTransaction(
            app_type='phonepe',
            amount=500.0,
            sender_name='John Doe',
            sender_upi='john@okhdfcbank',
            receiver_name='Jane Smith',
            receiver_upi='jane@oksbi',
            transaction_id='TXN123456',
            timestamp='2024-01-15 14:30:00',
            status='Completed',
            bank_name='HDFC Bank'
        )

        df = txn.to_features()

        assert len(df) == 1
        assert df['TransactionAmt'].iloc[0] == 500.0
        assert 'TransactionDT' in df.columns
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert df['hour'].iloc[0] == 14
        assert df['day_of_week'].iloc[0] == 0  # Monday

    def test_to_features_without_timestamp(self):
        txn = ParsedTransaction(
            app_type='phonepe',
            amount=1000.0,
        )

        df = txn.to_features()

        assert df['TransactionAmt'].iloc[0] == 1000.0
        assert df['TransactionDT'].iloc[0] == 0
        assert df['hour'].iloc[0] == 0

    def test_hash_string(self):
        txn = ParsedTransaction(app_type='phonepe')
        assert txn._hash_string('test@upi') > 0
        assert txn._hash_string('') == 0

    def test_extract_domain(self):
        txn = ParsedTransaction(app_type='phonepe')
        assert txn._extract_domain('user@okhdfcbank') == 'okhdfcbank'
        assert txn._extract_domain('user@sbi') == 'sbi'
        assert txn._extract_domain('noat') == 'unknown'


class TestScreenshotParser:
    """Tests for ScreenshotParser class."""

    @pytest.fixture
    def parser(self):
        return ScreenshotParser()

    def test_detect_app_phonepe(self, parser):
        text = "PhonePe Payment Successful\nFrom: John\nTo: Jane"
        assert parser.detect_app(text) == 'phonepe'

    def test_detect_app_gpay(self, parser):
        text = "Google Pay Transaction\nSent by: John\nPaid to: Jane"
        assert parser.detect_app(text) == 'gpay'

    def test_detect_app_gpay_variants(self, parser):
        for text in ["GPay", "tez payment", "GooglePay"]:
            assert parser.detect_app(text) == 'gpay'

    def test_detect_app_default(self, parser):
        text = "Some random text without app name"
        assert parser.detect_app(text) == 'phonepe'

    def test_extract_amount(self, parser):
        text = "Amount: ₹1,500.00"
        amount = parser._extract_amount(text, parser.PHONEPE_PATTERNS)
        assert amount == 1500.0

    def test_extract_amount_multiple(self, parser):
        text = "₹500 sent, ₹1,000 received"
        amount = parser._extract_amount(text, parser.PHONEPE_PATTERNS)
        assert amount == 1000.0  # Should pick the largest

    def test_extract_first_match(self, parser):
        text = "From: John Doe\nTo: Jane Smith"
        result = parser._extract_first_match(text, [r'From\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)'])
        assert result == 'John Doe'

    def test_extract_first_match_no_match(self, parser):
        text = "No matching pattern here"
        result = parser._extract_first_match(text, [r'From\s*[:\-]?\s*([A-Za-z\s]+)'])
        assert result is None


class TestScreenshotParserMocked:
    """Tests with mocked OCR to avoid needing actual images."""

    @pytest.fixture
    def parser(self):
        return ScreenshotParser()

    def test_parse_phonepe_mocked(self, parser):
        # Mock OCR output
        mock_text = """
        PhonePe
        Payment Successful
        Amount: ₹1,500.00
        From: John Doe
        UPI: john@okhdfcbank
        To: Jane Smith
        UPI: jane@oksbi
        Transaction ID: TXN123456789
        Date: 15/01/2024, 14:30 PM
        Status: Completed
        Bank: HDFC Bank
        """

        with patch.object(parser, 'extract_text', return_value=mock_text):
            with patch('src.screenshot_parser.Image.open'):
                with patch('src.screenshot_parser.pytesseract.image_to_string', return_value=mock_text):
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        result = parser.parse(Path(tmp.name))

        assert result.app_type == 'phonepe'
        assert result.amount == 1500.0
        assert result.sender_name == 'John Doe'
        assert result.sender_upi == 'john@okhdfcbank'
        assert result.receiver_name == 'Jane Smith'
        assert result.receiver_upi == 'jane@oksbi'
        assert result.transaction_id == 'TXN123456789'
        assert 'Completed' in result.status or 'Successful' in result.status
        assert 'HDFC' in result.bank_name

    def test_parse_gpay_mocked(self, parser):
        mock_text = """
        Google Pay
        Transaction Complete
        ₹500.00
        Sent by: Alice
        UPI: alice@okaxis
        Paid to: Bob
        UPI: bob@okicici
        GPAY: GP123456
        Jan 15, 2024 2:30 PM
        Status: Success
        Via: ICICI Bank
        """

        with patch.object(parser, 'extract_text', return_value=mock_text):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                result = parser.parse(Path(tmp.name))

        assert result.app_type == 'gpay'
        assert result.amount == 500.0
        assert result.sender_name == 'Alice'
        assert result.sender_upi == 'alice@okaxis'
        assert result.receiver_name == 'Bob'
        assert result.receiver_upi == 'bob@okicici'
        assert result.transaction_id == 'GP123456'


class TestParseScreenshotFunction:
    """Tests for the convenience parse_screenshot function."""

    def test_parse_screenshot_error_no_ocr(self):
        # This test only runs if OCR is not available
        if OCR_AVAILABLE:
            pytest.skip("OCR available, skipping error test")

        with pytest.raises(ScreenshotParserError):
            parse_screenshot('dummy.png')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])